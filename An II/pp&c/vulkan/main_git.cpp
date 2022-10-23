#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include <iostream>

#define WIDTH 28
#define HEIGHT 28
#define CHANNELS 16

#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal: VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

typedef struct ComputeApp {
    VkInstance instance;
    VkDebugReportCallbackEXT debugReportCallback;
    VkBuffer inputBuffer;
    VkBuffer outputBuffer;
    VkDeviceMemory bufferMemory;
    unsigned bufferSize;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet[2];
    VkDescriptorSetLayout descriptorSetLayout;
    const char * enabledLayers[10];
    unsigned enabledLayersCount;
    VkQueue queue;
    uint32_t queueFamilyIndex;
} ComputeApp;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

    printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

    return VK_FALSE;
}

void createInstance(ComputeApp * computeApp) {
    const char * enabledExtensions;

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, NULL);

    VkLayerProperties layerProperties[20];
    vkEnumerateInstanceLayerProperties(&layerCount, layerProperties);

    unsigned foundLayer = 0;
    for (unsigned i = 0; i < layerCount; i++) {

        if (strcmp("VK_LAYER_LUNARG_standard_validation", layerProperties[i].layerName) == 0) {
            foundLayer = 1;
            break;
        }

    }

    if (!foundLayer) {
        printf("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
        exit(1);
    }

    computeApp->enabledLayers[computeApp->enabledLayersCount] = "VK_LAYER_LUNARG_standard_validation";
    computeApp->enabledLayersCount++;


    uint32_t extensionCount;

    vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
    VkExtensionProperties extensionProperties[100];
    vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties);

    unsigned foundExtension = 0;
    for (unsigned i = 0; i < extensionCount; i++) {
        if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, extensionProperties[i].extensionName) == 0) {
            foundExtension = 1;
            break;
        }
    }

    if (!foundExtension) {
        printf("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
        exit(1);
    }
    enabledExtensions = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;


    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "ComputeApp";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "compute_app";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    createInfo.enabledLayerCount = computeApp->enabledLayersCount;
    createInfo.ppEnabledLayerNames = computeApp->enabledLayers;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = &enabledExtensions;

    VK_CHECK_RESULT(vkCreateInstance(&createInfo, NULL, &computeApp->instance));

    VkDebugReportCallbackCreateInfoEXT createInfoDebug = {};
    createInfoDebug.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    createInfoDebug.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    createInfoDebug.pfnCallback = &debugReportCallbackFn;

    PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(computeApp->instance, "vkCreateDebugReportCallbackEXT");
    if (vkCreateDebugReportCallbackEXT == NULL) {
        printf("Could not load vkCreateDebugReportCallbackEXT");
        exit(1);
    }

    VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(computeApp->instance, &createInfoDebug, NULL, &computeApp->debugReportCallback));
}

void findPhysicalDevice(ComputeApp * computeApp) {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(computeApp->instance, &deviceCount, NULL);

    if (deviceCount == 0) {
        printf("could not find a device with vulkan support");
        exit(1);
    }

    VkPhysicalDevice devices[10];
    vkEnumeratePhysicalDevices(computeApp->instance, &deviceCount, devices);

    computeApp->physicalDevice = devices[0];
}

void getComputeQueueFamilyIndex(ComputeApp * computeApp) {
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(computeApp->physicalDevice, &queueFamilyCount, NULL);

    VkQueueFamilyProperties queueFamilies[10];
    vkGetPhysicalDeviceQueueFamilyProperties(computeApp->physicalDevice, &queueFamilyCount, queueFamilies);

    uint32_t i = 0;
    for (; i < queueFamilyCount; ++i) {
        VkQueueFamilyProperties props = queueFamilies[i];

        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            break;
        }
    }

    if (i == queueFamilyCount) {
        printf("could not find a queue family that supports operations\n");
    }

    computeApp->queueFamilyIndex = i;
}

void createDevice(ComputeApp * computeApp) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;

    queueCreateInfo.queueFamilyIndex = computeApp->queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriorities = 1.0;
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    VkDeviceCreateInfo deviceCreateInfo = {};

    VkPhysicalDeviceFeatures deviceFeatures = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = computeApp->enabledLayersCount;
    deviceCreateInfo.ppEnabledLayerNames = computeApp->enabledLayers;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VK_CHECK_RESULT(vkCreateDevice(computeApp->physicalDevice, &deviceCreateInfo, NULL, &computeApp->device)); // create logical device.

    vkGetDeviceQueue(computeApp->device, computeApp->queueFamilyIndex, 0, &computeApp->queue);
}

uint32_t findMemoryType(ComputeApp * computeApp, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(computeApp->physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }

    printf("findMemoryType: Memory type does not found.\n");
    exit(1);
}

void createBuffer(ComputeApp * computeApp) {
    VkBufferCreateInfo inBufferCreateInfo = {};
    inBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    inBufferCreateInfo.size = computeApp->bufferSize;
    inBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    inBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(computeApp->device, &inBufferCreateInfo, NULL, &computeApp->inputBuffer));

    VkBufferCreateInfo outBufferCreateInfo = {};
    outBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outBufferCreateInfo.size = computeApp->bufferSize;
    outBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    outBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(computeApp->device, &outBufferCreateInfo, NULL, &computeApp->outputBuffer));

    VkMemoryRequirements inputMemoryRequirements;
    vkGetBufferMemoryRequirements(computeApp->device, computeApp->inputBuffer, &inputMemoryRequirements);

    VkMemoryRequirements outputMemoryRequirements;
    vkGetBufferMemoryRequirements(computeApp->device, computeApp->outputBuffer, &outputMemoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = computeApp->bufferSize * 2;

    allocateInfo.memoryTypeIndex = findMemoryType(
            computeApp,
            inputMemoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );

    VK_CHECK_RESULT(vkAllocateMemory(computeApp->device, &allocateInfo, NULL, &computeApp->bufferMemory)); // allocate memory on device.

    VK_CHECK_RESULT(vkBindBufferMemory(computeApp->device, computeApp->inputBuffer, computeApp->bufferMemory, 0));

    VK_CHECK_RESULT(vkBindBufferMemory(computeApp->device, computeApp->outputBuffer, computeApp->bufferMemory, computeApp->bufferSize));
}

void createDescriptorSetLayout(ComputeApp * computeApp) {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[2] = {};

    descriptorSetLayoutBinding[0].binding = 0;
    descriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[0].descriptorCount = 1;
    descriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBinding[1].binding = 1;
    descriptorSetLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[1].descriptorCount = 1;
    descriptorSetLayoutBinding[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 2; // only a single binding in this descriptor set layout.
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding;

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(computeApp->device, &descriptorSetLayoutCreateInfo, NULL, &computeApp->descriptorSetLayout));
}

void createDescriptorSet(ComputeApp * computeApp) {
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    // create descriptor pool.
    VK_CHECK_RESULT(vkCreateDescriptorPool(computeApp->device, &descriptorPoolCreateInfo, NULL, &computeApp->descriptorPool));


    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = computeApp->descriptorPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &computeApp->descriptorSetLayout;


    // allocate descriptor set.
    VK_CHECK_RESULT(vkAllocateDescriptorSets(computeApp->device, &descriptorSetAllocateInfo, &computeApp->descriptorSet[0]));

    // Specify the buffer to bind to the descriptor.
    VkDescriptorBufferInfo inputDescriptorBufferInfo = {};
    inputDescriptorBufferInfo.buffer = computeApp->inputBuffer;
    inputDescriptorBufferInfo.offset = 0;
    inputDescriptorBufferInfo.range = computeApp->bufferSize;

    VkDescriptorBufferInfo outputDescriptorBufferInfo = {};
    outputDescriptorBufferInfo.buffer = computeApp->outputBuffer;
    outputDescriptorBufferInfo.offset = 0;
    outputDescriptorBufferInfo.range = computeApp->bufferSize;

    VkWriteDescriptorSet writeDescriptorSet[2] = {};

    writeDescriptorSet[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet = computeApp->descriptorSet[0]; // write to this descriptor set.
    writeDescriptorSet[0].dstBinding = 0;
    writeDescriptorSet[0].descriptorCount = 1;
    writeDescriptorSet[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    writeDescriptorSet[0].pBufferInfo = &inputDescriptorBufferInfo;

    writeDescriptorSet[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet = computeApp->descriptorSet[0]; // write to this descriptor set.
    writeDescriptorSet[1].dstBinding = 1;
    writeDescriptorSet[1].descriptorCount = 1;
    writeDescriptorSet[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    writeDescriptorSet[1].pBufferInfo = &outputDescriptorBufferInfo;

    vkUpdateDescriptorSets(computeApp->device, 2, writeDescriptorSet, 0, NULL);
}

void createShaderModule(ComputeApp * computeApp, const char * filename) {
    uint32_t filelength;

    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error: Shader spv file not found - %s\n", filename);
        exit(1);
    }

    // get file size.
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    long filesizepadded = (long)(ceil(filesize / 4.0) * 4);

    // read file contents.
    /*
     *char *str = malloc((size_t)filesizepadded);
     */
    char* str = new char[filesizepadded];
    fread(str, filesize, sizeof(char), fp);
    fclose(fp);

    // data padding.
    for (long i = filesize; i < filesizepadded; i++) {
        str[i] = 0;
    }

    filelength = filesizepadded;
    uint32_t* code = (uint32_t *)str;

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = filelength;

    VK_CHECK_RESULT(vkCreateShaderModule(computeApp->device, &createInfo, NULL, &computeApp->computeShaderModule));
    free(code);
}

void createComputePipeline(ComputeApp * computeApp) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeApp->computeShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &computeApp->descriptorSetLayout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(computeApp->device, &pipelineLayoutCreateInfo, NULL, &computeApp->pipelineLayout));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = computeApp->pipelineLayout;

    VK_CHECK_RESULT(vkCreateComputePipelines(
            computeApp->device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &computeApp->pipeline));
}

void createCommandBuffer(ComputeApp * computeApp) {
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = computeApp->queueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(computeApp->device, &commandPoolCreateInfo, NULL, &computeApp->commandPool));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = computeApp->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(computeApp->device, &commandBufferAllocateInfo, &computeApp->commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(computeApp->commandBuffer, &beginInfo));

    vkCmdBindPipeline(computeApp->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeApp->pipeline);
    vkCmdBindDescriptorSets(computeApp->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeApp->pipelineLayout, 0, 1, &computeApp->descriptorSet[0], 0, NULL);

    vkCmdDispatch(computeApp->commandBuffer, (uint32_t)ceil(WIDTH / 7.), (uint32_t)(HEIGHT / 7.0), 1);

    VK_CHECK_RESULT(vkEndCommandBuffer(computeApp->commandBuffer));
}

void setInputBuffer(ComputeApp *computeApp) {
    float * data = 0;
    VkMemoryMapFlags memoryFlags = 0;

    VK_CHECK_RESULT(vkMapMemory(
            computeApp->device,
            computeApp->bufferMemory,
            0,
            computeApp->bufferSize,
            memoryFlags,
            (void **)&data
    ));

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
          for (int z = 0; z < CHANNELS; z++) {
            data[z * WIDTH * HEIGHT + i * WIDTH + j] = 2;
          }
        }
    }

    vkUnmapMemory(computeApp->device, computeApp->bufferMemory);

    float * outData = 0;
    VK_CHECK_RESULT(vkMapMemory(
            computeApp->device,
            computeApp->bufferMemory,
            computeApp->bufferSize,
            computeApp->bufferSize,
            memoryFlags,
            (void **)&outData
    ));

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
          for (int z = 0; z < CHANNELS; z++) {
            outData[z * WIDTH * HEIGHT + i * WIDTH + j] = 0;
          }
        }
    }

    vkUnmapMemory(computeApp->device, computeApp->bufferMemory);
}

void runCommandBuffer(ComputeApp * computeApp) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeApp->commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;

    VK_CHECK_RESULT(vkCreateFence(computeApp->device, &fenceCreateInfo, NULL, &fence));
    VK_CHECK_RESULT(vkQueueSubmit(computeApp->queue, 1, &submitInfo, fence));
    VK_CHECK_RESULT(vkWaitForFences(computeApp->device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(computeApp->device, fence, NULL);
}

void cleanup(ComputeApp * computeApp) {
    PFN_vkDestroyDebugReportCallbackEXT func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(computeApp->instance, "vkDestroyDebugReportCallbackEXT");
    if (func == NULL) {
       printf("Could not load vkDestroyDebugReportCallbackEXT");
    }

    func(computeApp->instance, computeApp->debugReportCallback, NULL);

    vkFreeMemory(computeApp->device, computeApp->bufferMemory, NULL);
    vkDestroyBuffer(computeApp->device, computeApp->inputBuffer, NULL);
    vkDestroyBuffer(computeApp->device, computeApp->outputBuffer, NULL);

    vkDestroyShaderModule(computeApp->device, computeApp->computeShaderModule, NULL);
    vkDestroyDescriptorPool(computeApp->device, computeApp->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(computeApp->device, computeApp->descriptorSetLayout, NULL);
    vkDestroyPipelineLayout(computeApp->device, computeApp->pipelineLayout, NULL);
    vkDestroyPipeline(computeApp->device, computeApp->pipeline, NULL);
    vkDestroyCommandPool(computeApp->device, computeApp->commandPool, NULL);
    vkDestroyDevice(computeApp->device, NULL);
    vkDestroyInstance(computeApp->instance, NULL);
}

int main() {
    ComputeApp computeApp;
    computeApp.bufferSize = sizeof(float) * WIDTH * HEIGHT * CHANNELS;
    computeApp.enabledLayersCount = 0;

    createInstance(&computeApp);
    findPhysicalDevice(&computeApp);
    getComputeQueueFamilyIndex(&computeApp);
    createDevice(&computeApp);
    createBuffer(&computeApp);
    createDescriptorSetLayout(&computeApp);
    createDescriptorSet(&computeApp);
    createShaderModule(&computeApp, "shaders/comp.spv");
    createComputePipeline(&computeApp);
    createCommandBuffer(&computeApp);
    setInputBuffer(&computeApp);

    runCommandBuffer(&computeApp);

    float * result = 0;
    VkMemoryMapFlags memoryFlags = 0;

    VK_CHECK_RESULT(vkMapMemory(
        computeApp.device,
        computeApp.bufferMemory,
        computeApp.bufferSize,
        computeApp.bufferSize,
        memoryFlags,
        (void **)&result
    ));


    printf("Output Result: \n");
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        printf("%2f ", result[i * WIDTH + j]);
      }
      printf("\n");
    }
    printf("\n");
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        printf("%2f ", result[1 * WIDTH * HEIGHT + i * WIDTH + j]);
      }
      printf("\n");
    }
    printf("\n");


    vkUnmapMemory(computeApp.device, computeApp.bufferMemory);

    cleanup(&computeApp);

    return 0;
}
