const std = @import("std");
const types = @import("types.zig");

pub const VkBufferCopy = extern struct {
    srcOffset: u64 = 0,
    dstOffset: u64 = 0,
    size: u64 = 0,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: types.VkStructureType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: [*]const VkDescriptorSetLayoutBinding,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: u32,
    offset: u32,
    size: u32,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: types.VkStructureType = .PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32,
    pSetLayouts: [*]const types.VkDescriptorSetLayout,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?[*]const VkPushConstantRange = null,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: types.VkStructureType = .SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: types.VkStructureType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32 = types.VK_SHADER_STAGE_COMPUTE_BIT,
    module: types.VkShaderModule,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*const anyopaque = null,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: types.VkStructureType = .COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: types.VkPipelineLayout,
    basePipelineHandle: types.VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

pub const VkDescriptorPoolSize = extern struct {
    descriptorType: u32,
    descriptorCount: u32,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: types.VkStructureType = .DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: [*]const VkDescriptorPoolSize,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: types.VkStructureType = .DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: types.VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: [*]const types.VkDescriptorSetLayout,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: types.VkBuffer,
    offset: u64,
    range: u64,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: types.VkStructureType = .WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: types.VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32 = 0,
    descriptorCount: u32,
    descriptorType: u32,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?[*]const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: types.VkStructureType = .COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 2, // RESET_COMMAND_BUFFER_BIT
    queueFamilyIndex: u32,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: types.VkStructureType = .COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: types.VkCommandPool,
    level: u32 = 0, // PRIMARY
    commandBufferCount: u32,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: types.VkStructureType = .COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = types.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const VkMemoryBarrier = extern struct {
    sType: types.VkStructureType = .MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32 = types.VK_ACCESS_SHADER_WRITE_BIT,
    dstAccessMask: u32 = types.VK_ACCESS_SHADER_READ_BIT | types.VK_ACCESS_SHADER_WRITE_BIT,
};

pub const VkBufferMemoryBarrier = extern struct {
    sType: types.VkStructureType = .BUFFER_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32,
    dstAccessMask: u32,
    srcQueueFamilyIndex: u32 = 0xFFFFFFFF,
    dstQueueFamilyIndex: u32 = 0xFFFFFFFF,
    buffer: types.VkBuffer,
    offset: u64 = 0,
    size: u64 = 0xFFFFFFFFFFFFFFFF,
};

pub const VkSubmitInfo = extern struct {
    sType: types.VkStructureType = .SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const u32 = null,
    commandBufferCount: u32,
    pCommandBuffers: [*]const types.VkCommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: types.VkStructureType = .FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};
