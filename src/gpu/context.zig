const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const vk_api = @import("vk_api.zig");

pub const GpuContext = struct {
    allocator: std.mem.Allocator,
    api: vk_api.VulkanApi,
    instance: types.VkInstance,
    physical_device: types.VkPhysicalDevice,
    device: types.VkDevice,
    queue: types.VkQueue,
    queue_family_index: u32,
    mem_props: types.VkPhysicalDeviceMemoryProperties,
    device_name: [256]u8,

    pub fn init(allocator: std.mem.Allocator) !GpuContext {
        var api = try vk_api.VulkanApi.load();
        errdefer api.deinit();

        const app_info = types.VkApplicationInfo{};
        const inst_info = types.VkInstanceCreateInfo{ .pApplicationInfo = &app_info };
        var instance: types.VkInstance = null;
        if (api.vkCreateInstance(&inst_info, null, &instance) != .SUCCESS) return error.VkInstanceCreationFailed;
        errdefer api.vkDestroyInstance(instance, null);

        var num_devices: u32 = 0;
        _ = api.vkEnumeratePhysicalDevices(instance, &num_devices, null);
        if (num_devices == 0) return error.NoVulkanDevices;

        const pdevices = try allocator.alloc(types.VkPhysicalDevice, num_devices);
        defer allocator.free(pdevices);
        _ = api.vkEnumeratePhysicalDevices(instance, &num_devices, pdevices.ptr);

        // Pick AMD device or first GPU
        var selected_device: ?types.VkPhysicalDevice = null;
        var selected_props: types.VkPhysicalDeviceProperties = undefined;
        var selected_q_family: ?u32 = null;

        for (pdevices) |pdev| {
            var props: types.VkPhysicalDeviceProperties = undefined;
            api.vkGetPhysicalDeviceProperties(pdev, &props);

            var num_q: u32 = 0;
            api.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &num_q, null);
            const q_props = try allocator.alloc(types.VkQueueFamilyProperties, num_q);
            defer allocator.free(q_props);
            api.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &num_q, q_props.ptr);

            var dedicated_compute_q: ?u32 = null;
            var fallback_compute_q: ?u32 = null;
            for (q_props, 0..) |qp, idx| {
                if ((qp.queueFlags & types.VK_QUEUE_COMPUTE_BIT) != 0) {
                    if ((qp.queueFlags & types.VK_QUEUE_GRAPHICS_BIT) == 0 and dedicated_compute_q == null) {
                        dedicated_compute_q = @intCast(idx);
                    } else if (fallback_compute_q == null) {
                        fallback_compute_q = @intCast(idx);
                    }
                }
            }
            const compute_q = dedicated_compute_q orelse fallback_compute_q;

            if (compute_q) |cq| {
                if (selected_device == null or props.vendorID == 0x1002) {
                    selected_device = pdev;
                    selected_props = props;
                    selected_q_family = cq;
                    if (props.vendorID == 0x1002) break; // Prefer AMD GPU
                }
            }
        }

        const pdev = selected_device orelse return error.NoComputeDeviceFound;
        const q_family = selected_q_family.?;

        const priority: f32 = 1.0;
        const q_create = types.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = q_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        const dev_create = types.VkDeviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = (&q_create)[0..1].ptr,
        };
        var device: types.VkDevice = null;
        if (api.vkCreateDevice(pdev, &dev_create, null, &device) != .SUCCESS) return error.VkDeviceCreationFailed;
        errdefer api.vkDestroyDevice(device, null);

        var queue: types.VkQueue = null;
        api.vkGetDeviceQueue(device, q_family, 0, &queue);

        var mem_props: types.VkPhysicalDeviceMemoryProperties = undefined;
        api.vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);

        return .{
            .allocator = allocator,
            .api = api,
            .instance = instance,
            .physical_device = pdev,
            .device = device,
            .queue = queue,
            .queue_family_index = q_family,
            .mem_props = mem_props,
            .device_name = selected_props.deviceName,
        };
    }

    pub fn deinit(self: *GpuContext) void {
        self.api.vkDestroyDevice(self.device, null);
        self.api.vkDestroyInstance(self.instance, null);
        self.api.deinit();
    }

    pub fn findMemoryType(self: *const GpuContext, type_filter: u32, properties: u32) !u32 {
        for (0..self.mem_props.memoryTypeCount) |i| {
            const idx: u32 = @intCast(i);
            const matches_filter = (type_filter & (@as(u32, 1) << @truncate(idx))) != 0;
            const matches_props = (self.mem_props.memoryTypes[i].propertyFlags & properties) == properties;
            if (matches_filter and matches_props) return idx;
        }
        // Fallback: relax properties if HOST_COHERENT not found
        for (0..self.mem_props.memoryTypeCount) |i| {
            const idx: u32 = @intCast(i);
            const matches_filter = (type_filter & (@as(u32, 1) << @truncate(idx))) != 0;
            if (matches_filter and (self.mem_props.memoryTypes[i].propertyFlags & types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                return idx;
            }
        }
        return error.NoMatchingMemoryType;
    }
};
