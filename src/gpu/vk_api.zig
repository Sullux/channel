const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");

pub const PFN_vkCreateInstance = *const fn (*const types.VkInstanceCreateInfo, ?*const anyopaque, *types.VkInstance) callconv(.C) types.VkResult;
pub const PFN_vkDestroyInstance = *const fn (types.VkInstance, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkEnumeratePhysicalDevices = *const fn (types.VkInstance, *u32, ?[*]types.VkPhysicalDevice) callconv(.C) types.VkResult;
pub const PFN_vkGetPhysicalDeviceProperties = *const fn (types.VkPhysicalDevice, *types.VkPhysicalDeviceProperties) callconv(.C) void;
pub const PFN_vkGetPhysicalDeviceMemoryProperties = *const fn (types.VkPhysicalDevice, *types.VkPhysicalDeviceMemoryProperties) callconv(.C) void;
pub const PFN_vkGetPhysicalDeviceQueueFamilyProperties = *const fn (types.VkPhysicalDevice, *u32, ?[*]types.VkQueueFamilyProperties) callconv(.C) void;
pub const PFN_vkCreateDevice = *const fn (types.VkPhysicalDevice, *const types.VkDeviceCreateInfo, ?*const anyopaque, *types.VkDevice) callconv(.C) types.VkResult;
pub const PFN_vkDestroyDevice = *const fn (types.VkDevice, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkGetDeviceQueue = *const fn (types.VkDevice, u32, u32, *types.VkQueue) callconv(.C) void;
pub const PFN_vkCreateBuffer = *const fn (types.VkDevice, *const types.VkBufferCreateInfo, ?*const anyopaque, *types.VkBuffer) callconv(.C) types.VkResult;
pub const PFN_vkDestroyBuffer = *const fn (types.VkDevice, types.VkBuffer, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkGetBufferMemoryRequirements = *const fn (types.VkDevice, types.VkBuffer, *types.VkMemoryRequirements) callconv(.C) void;
pub const PFN_vkAllocateMemory = *const fn (types.VkDevice, *const types.VkMemoryAllocateInfo, ?*const anyopaque, *types.VkDeviceMemory) callconv(.C) types.VkResult;
pub const PFN_vkFreeMemory = *const fn (types.VkDevice, types.VkDeviceMemory, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkBindBufferMemory = *const fn (types.VkDevice, types.VkBuffer, types.VkDeviceMemory, u64) callconv(.C) types.VkResult;
pub const PFN_vkMapMemory = *const fn (types.VkDevice, types.VkDeviceMemory, u64, u64, u32, *?*anyopaque) callconv(.C) types.VkResult;
pub const PFN_vkUnmapMemory = *const fn (types.VkDevice, types.VkDeviceMemory) callconv(.C) void;
pub const PFN_vkCreateShaderModule = *const fn (types.VkDevice, *const types_dispatch.VkShaderModuleCreateInfo, ?*const anyopaque, *types.VkShaderModule) callconv(.C) types.VkResult;
pub const PFN_vkDestroyShaderModule = *const fn (types.VkDevice, types.VkShaderModule, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCreateDescriptorSetLayout = *const fn (types.VkDevice, *const types_dispatch.VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *types.VkDescriptorSetLayout) callconv(.C) types.VkResult;
pub const PFN_vkDestroyDescriptorSetLayout = *const fn (types.VkDevice, types.VkDescriptorSetLayout, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCreatePipelineLayout = *const fn (types.VkDevice, *const types_dispatch.VkPipelineLayoutCreateInfo, ?*const anyopaque, *types.VkPipelineLayout) callconv(.C) types.VkResult;
pub const PFN_vkDestroyPipelineLayout = *const fn (types.VkDevice, types.VkPipelineLayout, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCreateComputePipelines = *const fn (types.VkDevice, ?*anyopaque, u32, [*]const types_dispatch.VkComputePipelineCreateInfo, ?*const anyopaque, [*]types.VkPipeline) callconv(.C) types.VkResult;
pub const PFN_vkDestroyPipeline = *const fn (types.VkDevice, types.VkPipeline, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCreateDescriptorPool = *const fn (types.VkDevice, *const types_dispatch.VkDescriptorPoolCreateInfo, ?*const anyopaque, *types.VkDescriptorPool) callconv(.C) types.VkResult;
pub const PFN_vkDestroyDescriptorPool = *const fn (types.VkDevice, types.VkDescriptorPool, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkResetDescriptorPool = *const fn (types.VkDevice, types.VkDescriptorPool, u32) callconv(.C) types.VkResult;
pub const PFN_vkAllocateDescriptorSets = *const fn (types.VkDevice, *const types_dispatch.VkDescriptorSetAllocateInfo, [*]types.VkDescriptorSet) callconv(.C) types.VkResult;
pub const PFN_vkUpdateDescriptorSets = *const fn (types.VkDevice, u32, [*]const types_dispatch.VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCreateCommandPool = *const fn (types.VkDevice, *const types_dispatch.VkCommandPoolCreateInfo, ?*const anyopaque, *types.VkCommandPool) callconv(.C) types.VkResult;
pub const PFN_vkDestroyCommandPool = *const fn (types.VkDevice, types.VkCommandPool, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkAllocateCommandBuffers = *const fn (types.VkDevice, *const types_dispatch.VkCommandBufferAllocateInfo, [*]types.VkCommandBuffer) callconv(.C) types.VkResult;
pub const PFN_vkBeginCommandBuffer = *const fn (types.VkCommandBuffer, *const types_dispatch.VkCommandBufferBeginInfo) callconv(.C) types.VkResult;
pub const PFN_vkEndCommandBuffer = *const fn (types.VkCommandBuffer) callconv(.C) types.VkResult;
pub const PFN_vkCmdBindPipeline = *const fn (types.VkCommandBuffer, u32, types.VkPipeline) callconv(.C) void;
pub const PFN_vkCmdBindDescriptorSets = *const fn (types.VkCommandBuffer, u32, types.VkPipelineLayout, u32, u32, [*]const types.VkDescriptorSet, u32, ?*const u32) callconv(.C) void;
pub const PFN_vkCmdPushConstants = *const fn (types.VkCommandBuffer, types.VkPipelineLayout, u32, u32, u32, *const anyopaque) callconv(.C) void;
pub const PFN_vkCmdCopyBuffer = *const fn (types.VkCommandBuffer, types.VkBuffer, types.VkBuffer, u32, [*]const types_dispatch.VkBufferCopy) callconv(.C) void;
pub const PFN_vkCmdPipelineBarrier = *const fn (types.VkCommandBuffer, u32, u32, u32, u32, ?*const anyopaque, u32, ?[*]const types_dispatch.VkBufferMemoryBarrier, u32, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkCmdDispatch = *const fn (types.VkCommandBuffer, u32, u32, u32) callconv(.C) void;
pub const PFN_vkCmdDispatchIndirect = *const fn (types.VkCommandBuffer, types.VkBuffer, u64) callconv(.C) void;
pub const PFN_vkCreateFence = *const fn (types.VkDevice, *const types_dispatch.VkFenceCreateInfo, ?*const anyopaque, *types.VkFence) callconv(.C) types.VkResult;
pub const PFN_vkDestroyFence = *const fn (types.VkDevice, types.VkFence, ?*const anyopaque) callconv(.C) void;
pub const PFN_vkResetFences = *const fn (types.VkDevice, u32, [*]const types.VkFence) callconv(.C) types.VkResult;
pub const PFN_vkGetFenceStatus = *const fn (types.VkDevice, types.VkFence) callconv(.C) types.VkResult;
pub const PFN_vkWaitForFences = *const fn (types.VkDevice, u32, [*]const types.VkFence, u32, u64) callconv(.C) types.VkResult;
pub const PFN_vkQueueSubmit = *const fn (types.VkQueue, u32, [*]const types_dispatch.VkSubmitInfo, types.VkFence) callconv(.C) types.VkResult;
pub const PFN_vkQueueWaitIdle = *const fn (types.VkQueue) callconv(.C) types.VkResult;

pub const VulkanApi = struct {
    lib: std.DynLib,

    vkCreateInstance: PFN_vkCreateInstance,
    vkDestroyInstance: PFN_vkDestroyInstance,
    vkEnumeratePhysicalDevices: PFN_vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties: PFN_vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceMemoryProperties: PFN_vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceQueueFamilyProperties: PFN_vkGetPhysicalDeviceQueueFamilyProperties,
    vkCreateDevice: PFN_vkCreateDevice,
    vkDestroyDevice: PFN_vkDestroyDevice,
    vkGetDeviceQueue: PFN_vkGetDeviceQueue,
    vkCreateBuffer: PFN_vkCreateBuffer,
    vkDestroyBuffer: PFN_vkDestroyBuffer,
    vkGetBufferMemoryRequirements: PFN_vkGetBufferMemoryRequirements,
    vkAllocateMemory: PFN_vkAllocateMemory,
    vkFreeMemory: PFN_vkFreeMemory,
    vkBindBufferMemory: PFN_vkBindBufferMemory,
    vkMapMemory: PFN_vkMapMemory,
    vkUnmapMemory: PFN_vkUnmapMemory,
    vkCreateShaderModule: PFN_vkCreateShaderModule,
    vkDestroyShaderModule: PFN_vkDestroyShaderModule,
    vkCreateDescriptorSetLayout: PFN_vkCreateDescriptorSetLayout,
    vkDestroyDescriptorSetLayout: PFN_vkDestroyDescriptorSetLayout,
    vkCreatePipelineLayout: PFN_vkCreatePipelineLayout,
    vkDestroyPipelineLayout: PFN_vkDestroyPipelineLayout,
    vkCreateComputePipelines: PFN_vkCreateComputePipelines,
    vkDestroyPipeline: PFN_vkDestroyPipeline,
    vkCreateDescriptorPool: PFN_vkCreateDescriptorPool,
    vkDestroyDescriptorPool: PFN_vkDestroyDescriptorPool,
    vkResetDescriptorPool: PFN_vkResetDescriptorPool,
    vkAllocateDescriptorSets: PFN_vkAllocateDescriptorSets,
    vkUpdateDescriptorSets: PFN_vkUpdateDescriptorSets,
    vkCreateCommandPool: PFN_vkCreateCommandPool,
    vkDestroyCommandPool: PFN_vkDestroyCommandPool,
    vkAllocateCommandBuffers: PFN_vkAllocateCommandBuffers,
    vkBeginCommandBuffer: PFN_vkBeginCommandBuffer,
    vkEndCommandBuffer: PFN_vkEndCommandBuffer,
    vkCmdBindPipeline: PFN_vkCmdBindPipeline,
    vkCmdBindDescriptorSets: PFN_vkCmdBindDescriptorSets,
    vkCmdPushConstants: PFN_vkCmdPushConstants,
    vkCmdCopyBuffer: PFN_vkCmdCopyBuffer,
    vkCmdPipelineBarrier: PFN_vkCmdPipelineBarrier,
    vkCmdDispatch: PFN_vkCmdDispatch,
    vkCmdDispatchIndirect: PFN_vkCmdDispatchIndirect,
    vkCreateFence: PFN_vkCreateFence,
    vkDestroyFence: PFN_vkDestroyFence,
    vkResetFences: PFN_vkResetFences,
    vkGetFenceStatus: PFN_vkGetFenceStatus,
    vkWaitForFences: PFN_vkWaitForFences,
    vkQueueSubmit: PFN_vkQueueSubmit,
    vkQueueWaitIdle: PFN_vkQueueWaitIdle,

    pub fn load() !VulkanApi {
        const lib_names = [_][]const u8{ "libvulkan.so.1", "libvulkan.so" };
        var dyn_lib: ?std.DynLib = null;
        for (lib_names) |name| {
            if (std.DynLib.open(name)) |lib| {
                dyn_lib = lib;
                break;
            } else |_| {}
        }
        var lib = dyn_lib orelse return error.VulkanLibraryNotFound;
        errdefer lib.close();

        return .{
            .lib = lib,
            .vkCreateInstance = lib.lookup(PFN_vkCreateInstance, "vkCreateInstance") orelse return error.SymbolNotFound,
            .vkDestroyInstance = lib.lookup(PFN_vkDestroyInstance, "vkDestroyInstance") orelse return error.SymbolNotFound,
            .vkEnumeratePhysicalDevices = lib.lookup(PFN_vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices") orelse return error.SymbolNotFound,
            .vkGetPhysicalDeviceProperties = lib.lookup(PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties") orelse return error.SymbolNotFound,
            .vkGetPhysicalDeviceMemoryProperties = lib.lookup(PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties") orelse return error.SymbolNotFound,
            .vkGetPhysicalDeviceQueueFamilyProperties = lib.lookup(PFN_vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return error.SymbolNotFound,
            .vkCreateDevice = lib.lookup(PFN_vkCreateDevice, "vkCreateDevice") orelse return error.SymbolNotFound,
            .vkDestroyDevice = lib.lookup(PFN_vkDestroyDevice, "vkDestroyDevice") orelse return error.SymbolNotFound,
            .vkGetDeviceQueue = lib.lookup(PFN_vkGetDeviceQueue, "vkGetDeviceQueue") orelse return error.SymbolNotFound,
            .vkCreateBuffer = lib.lookup(PFN_vkCreateBuffer, "vkCreateBuffer") orelse return error.SymbolNotFound,
            .vkDestroyBuffer = lib.lookup(PFN_vkDestroyBuffer, "vkDestroyBuffer") orelse return error.SymbolNotFound,
            .vkGetBufferMemoryRequirements = lib.lookup(PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements") orelse return error.SymbolNotFound,
            .vkAllocateMemory = lib.lookup(PFN_vkAllocateMemory, "vkAllocateMemory") orelse return error.SymbolNotFound,
            .vkFreeMemory = lib.lookup(PFN_vkFreeMemory, "vkFreeMemory") orelse return error.SymbolNotFound,
            .vkBindBufferMemory = lib.lookup(PFN_vkBindBufferMemory, "vkBindBufferMemory") orelse return error.SymbolNotFound,
            .vkMapMemory = lib.lookup(PFN_vkMapMemory, "vkMapMemory") orelse return error.SymbolNotFound,
            .vkUnmapMemory = lib.lookup(PFN_vkUnmapMemory, "vkUnmapMemory") orelse return error.SymbolNotFound,
            .vkCreateShaderModule = lib.lookup(PFN_vkCreateShaderModule, "vkCreateShaderModule") orelse return error.SymbolNotFound,
            .vkDestroyShaderModule = lib.lookup(PFN_vkDestroyShaderModule, "vkDestroyShaderModule") orelse return error.SymbolNotFound,
            .vkCreateDescriptorSetLayout = lib.lookup(PFN_vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout") orelse return error.SymbolNotFound,
            .vkDestroyDescriptorSetLayout = lib.lookup(PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout") orelse return error.SymbolNotFound,
            .vkCreatePipelineLayout = lib.lookup(PFN_vkCreatePipelineLayout, "vkCreatePipelineLayout") orelse return error.SymbolNotFound,
            .vkDestroyPipelineLayout = lib.lookup(PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout") orelse return error.SymbolNotFound,
            .vkCreateComputePipelines = lib.lookup(PFN_vkCreateComputePipelines, "vkCreateComputePipelines") orelse return error.SymbolNotFound,
            .vkDestroyPipeline = lib.lookup(PFN_vkDestroyPipeline, "vkDestroyPipeline") orelse return error.SymbolNotFound,
            .vkCreateDescriptorPool = lib.lookup(PFN_vkCreateDescriptorPool, "vkCreateDescriptorPool") orelse return error.SymbolNotFound,
            .vkDestroyDescriptorPool = lib.lookup(PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool") orelse return error.SymbolNotFound,
            .vkResetDescriptorPool = lib.lookup(PFN_vkResetDescriptorPool, "vkResetDescriptorPool") orelse return error.SymbolNotFound,
            .vkAllocateDescriptorSets = lib.lookup(PFN_vkAllocateDescriptorSets, "vkAllocateDescriptorSets") orelse return error.SymbolNotFound,
            .vkUpdateDescriptorSets = lib.lookup(PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets") orelse return error.SymbolNotFound,
            .vkCreateCommandPool = lib.lookup(PFN_vkCreateCommandPool, "vkCreateCommandPool") orelse return error.SymbolNotFound,
            .vkDestroyCommandPool = lib.lookup(PFN_vkDestroyCommandPool, "vkDestroyCommandPool") orelse return error.SymbolNotFound,
            .vkAllocateCommandBuffers = lib.lookup(PFN_vkAllocateCommandBuffers, "vkAllocateCommandBuffers") orelse return error.SymbolNotFound,
            .vkBeginCommandBuffer = lib.lookup(PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer") orelse return error.SymbolNotFound,
            .vkEndCommandBuffer = lib.lookup(PFN_vkEndCommandBuffer, "vkEndCommandBuffer") orelse return error.SymbolNotFound,
            .vkCmdBindPipeline = lib.lookup(PFN_vkCmdBindPipeline, "vkCmdBindPipeline") orelse return error.SymbolNotFound,
            .vkCmdBindDescriptorSets = lib.lookup(PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets") orelse return error.SymbolNotFound,
            .vkCmdPushConstants = lib.lookup(PFN_vkCmdPushConstants, "vkCmdPushConstants") orelse return error.SymbolNotFound,
            .vkCmdCopyBuffer = lib.lookup(PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer") orelse return error.SymbolNotFound,
            .vkCmdPipelineBarrier = lib.lookup(PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier") orelse return error.SymbolNotFound,
            .vkCmdDispatch = lib.lookup(PFN_vkCmdDispatch, "vkCmdDispatch") orelse return error.SymbolNotFound,
            .vkCmdDispatchIndirect = lib.lookup(PFN_vkCmdDispatchIndirect, "vkCmdDispatchIndirect") orelse return error.SymbolNotFound,
            .vkCreateFence = lib.lookup(PFN_vkCreateFence, "vkCreateFence") orelse return error.SymbolNotFound,
            .vkDestroyFence = lib.lookup(PFN_vkDestroyFence, "vkDestroyFence") orelse return error.SymbolNotFound,
            .vkResetFences = lib.lookup(PFN_vkResetFences, "vkResetFences") orelse return error.SymbolNotFound,
            .vkGetFenceStatus = lib.lookup(PFN_vkGetFenceStatus, "vkGetFenceStatus") orelse return error.SymbolNotFound,
            .vkWaitForFences = lib.lookup(PFN_vkWaitForFences, "vkWaitForFences") orelse return error.SymbolNotFound,
            .vkQueueSubmit = lib.lookup(PFN_vkQueueSubmit, "vkQueueSubmit") orelse return error.SymbolNotFound,
            .vkQueueWaitIdle = lib.lookup(PFN_vkQueueWaitIdle, "vkQueueWaitIdle") orelse return error.SymbolNotFound,
        };
    }

    pub fn deinit(self: *VulkanApi) void {
        self.lib.close();
    }
};
