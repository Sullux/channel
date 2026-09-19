const std = @import("std");

pub const VkResult = enum(i32) {
    SUCCESS = 0,
    NOT_READY = 1,
    TIMEOUT = 2,
    EVENT_SET = 3,
    EVENT_RESET = 4,
    INCOMPLETE = 5,
    ERROR_OUT_OF_HOST_MEMORY = -1,
    ERROR_OUT_OF_DEVICE_MEMORY = -2,
    ERROR_INITIALIZATION_FAILED = -3,
    ERROR_DEVICE_LOST = -4,
    ERROR_MEMORY_MAP_FAILED = -5,
    ERROR_LAYER_NOT_PRESENT = -6,
    ERROR_EXTENSION_NOT_PRESENT = -7,
    ERROR_FEATURE_NOT_PRESENT = -8,
    ERROR_INCOMPATIBLE_DRIVER = -9,
    ERROR_TOO_MANY_OBJECTS = -10,
    ERROR_FORMAT_NOT_SUPPORTED = -11,
    ERROR_FRAGMENTED_POOL = -12,
    ERROR_UNKNOWN = -13,
    _,
};

pub const VkStructureType = enum(i32) {
    APPLICATION_INFO = 0,
    INSTANCE_CREATE_INFO = 1,
    DEVICE_QUEUE_CREATE_INFO = 2,
    DEVICE_CREATE_INFO = 3,
    SUBMIT_INFO = 4,
    MEMORY_ALLOCATE_INFO = 5,
    MAPPED_MEMORY_RANGE = 6,
    FENCE_CREATE_INFO = 8,
    BUFFER_CREATE_INFO = 12,
    SHADER_MODULE_CREATE_INFO = 16,
    PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    COMPUTE_PIPELINE_CREATE_INFO = 29,
    PIPELINE_LAYOUT_CREATE_INFO = 30,
    DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    DESCRIPTOR_POOL_CREATE_INFO = 33,
    DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    WRITE_DESCRIPTOR_SET = 35,
    COMMAND_POOL_CREATE_INFO = 39,
    COMMAND_BUFFER_ALLOCATE_INFO = 40,
    COMMAND_BUFFER_BEGIN_INFO = 42,
    BUFFER_MEMORY_BARRIER = 44,
    MEMORY_BARRIER = 46,
    _,
};

pub const VkPhysicalDeviceType = enum(u32) {
    OTHER = 0,
    INTEGRATED_GPU = 1,
    DISCRETE_GPU = 2,
    VIRTUAL_GPU = 3,
    CPU = 4,
    _,
};

pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x00000008;

pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: u32 = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: u32 = 0x00000100;

pub const VkDispatchIndirectCommand = extern struct {
    x: u32,
    y: u32,
    z: u32,
};

pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;
pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000040;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;

pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};
pub const VkBuffer = ?*opaque {};
pub const VkDeviceMemory = ?*opaque {};
pub const VkShaderModule = ?*opaque {};
pub const VkPipeline = ?*opaque {};
pub const VkPipelineLayout = ?*opaque {};
pub const VkDescriptorSetLayout = ?*opaque {};
pub const VkDescriptorPool = ?*opaque {};
pub const VkDescriptorSet = ?*opaque {};
pub const VkCommandPool = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkFence = ?*opaque {};

pub const VkApplicationInfo = extern struct {
    sType: VkStructureType = .APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = "Gemma4Streaming",
    applicationVersion: u32 = 1,
    pEngineName: ?[*:0]const u8 = "PureZigCompute",
    engineVersion: u32 = 1,
    apiVersion: u32 = (1 << 22) | (3 << 12), // Vulkan 1.3
};

pub const VkInstanceCreateInfo = extern struct {
    sType: VkStructureType = .INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const ?[*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const ?[*:0]const u8 = null,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: VkPhysicalDeviceType,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: [504]u8,
    sparseProperties: [20]u8,
};

pub const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

pub const VkMemoryHeap = extern struct {
    size: u64,
    flags: u32,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: [3]u32,
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: VkStructureType = .DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: *const f32,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: VkStructureType = .DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: [*]const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const ?[*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const ?[*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType = .BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64,
    usage: u32,
    sharingMode: u32 = 0, // EXCLUSIVE
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*const u32 = null,
};

pub const VkMemoryRequirements = extern struct { size: u64, alignment: u64, memoryTypeBits: u32 };
pub const VkMemoryAllocateInfo = extern struct { sType: VkStructureType = .MEMORY_ALLOCATE_INFO, pNext: ?*const anyopaque = null, allocationSize: u64, memoryTypeIndex: u32 };
