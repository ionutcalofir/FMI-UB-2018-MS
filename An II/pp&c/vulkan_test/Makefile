VULKAN_SDK_PATH = /home/ionutc/Documents/vulkan/1.2.131.1/x86_64
CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib -lvulkan

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest -g main.cpp lodepng.cpp $(LDFLAGS)

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest
