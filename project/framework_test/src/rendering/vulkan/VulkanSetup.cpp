//
// Created by samuel on 08/02/2021.
//

#include "VulkanSetup.h"

#include <SDL_vulkan.h>
#include <rendering/vulkan/helpers/VulkanDebug.h>
#include <util/selectors.h>

#include "util/fatal_error.h"

#define CHECK_SDL_ERROR(success) do { FATAL_ERROR_IF(!success, "SDL Error: %s\n", SDL_GetError()); } while(0);
#define CHECKED_SDL(X) ([&]{ SDL_bool success = (X); CHECK_SDL_ERROR(success); }())

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData){

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        return VK_FALSE;
    }

    auto severity = vk::to_string(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity));
    auto type = vk::to_string(vk::DebugUtilsMessageTypeFlagsEXT(messageType));

    auto print_to = (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? stderr : stdout;
//    fprintf(print_to, "VulkanDebug [%s] [%s]: [%s] %s [%d]\n",
//            type.c_str(), severity.c_str(),
//            pCallbackData->pMessageIdName, pCallbackData->pMessage, pCallbackData->messageIdNumber);
    fprintf(print_to, "%s\n", pCallbackData->pMessage);

    // VK_FALSE => don't stop the application
    return VK_FALSE;
}

VulkanSetup::VulkanSetup(vk::ApplicationInfo appInfo, Size<uint32_t> windowSize)
    : windowSize(windowSize) {

    window = SDL_CreateWindow(
            appInfo.pApplicationName,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            windowSize.x, windowSize.y,
        SDL_WINDOW_VULKAN
    );

    // Get SDL required extensions
    uint32_t extensionCount;
    // First find how many...
    CHECKED_SDL(SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr));
    // ...then allocate a buffer and get the values.
    auto extensionNames = std::vector<const char*>(extensionCount);
    CHECKED_SDL(SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensionNames.data()));
    // Now add the extra ones we want.
    extensionNames.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensionNames.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

    // Get list of requested Vulkan layers.
    auto layerNames = std::vector<const char *>();
    if (VulkanDebug) {
        layerNames.push_back("VK_LAYER_KHRONOS_validation");
        extensionNames.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Create the vulkan instance
    {
        printf("Creating Vulkan instance\nLayers\n");
        for (const auto layer : layerNames) {
            printf("\t%s\n", layer);
        }
        printf("Extensions\n");
        for (const auto extension : extensionNames) {
            printf("\t%s\n", extension);
        }

        auto createInfo = vk::InstanceCreateInfo(
                vk::InstanceCreateFlags(),
                &appInfo,
                layerNames.size(),
                layerNames.data(),
                extensionNames.size(),
                extensionNames.data()
        );
        instance = vk::createInstanceUnique(createInfo);

        // Create the dynamic Vulkan loader
        dynamicLoader.init(*instance, vkGetInstanceProcAddr);
    }

    // Create the debug message handler, if enabled
    if (VulkanDebug) {
        auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
                vk::DebugUtilsMessengerCreateFlagsEXT(),
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                &vulkanDebug,
                nullptr
        );

        debugMessenger = instance->createDebugUtilsMessengerEXTUnique(
                createInfo,
                nullptr,
                dynamicLoader
        );
    }

    // Create the SDL/Vulkan Surface, and select the format/present mode to use
    {
        VkSurfaceKHR rawSurface = nullptr;
        CHECKED_SDL(SDL_Vulkan_CreateSurface(window, *instance, &rawSurface));
        surface = vk::UniqueSurfaceKHR(rawSurface, *instance);
    }

    // Select the physical device,
    {
        auto requiredDeviceExtensions = std::vector<const char*>{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        };
        auto requiredDeviceExtensionsSet = std::set<std::string>(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());

        physicalDevice = selectAnySuitable<vk::PhysicalDevice>(
                instance->enumeratePhysicalDevices(),
                [this, &requiredDeviceExtensionsSet](vk::PhysicalDevice potentialDevice){
                    auto deviceProperties = potentialDevice.getProperties();
                    if (deviceProperties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu)
                        return false; // Only accept discrete GPUs

                    auto potentialQueueFamilies = VulkanQueueFamilies::getForDevice(potentialDevice, surface);
                    if (potentialQueueFamilies == std::nullopt)
                        return false; // Can't support all of the queues we want

                    // Check if any extensions would be missing.
                    auto availableExtensions = potentialDevice.enumerateDeviceExtensionProperties();
                    auto missingExtensions = requiredDeviceExtensionsSet;
                    for (const auto& extension : availableExtensions) {
                        // Use std::string so that it doesn't check for pointer equality
                        missingExtensions.erase(std::string(extension.extensionName));
                    }
                    if (!missingExtensions.empty())
                        return false; // Some extensions missing

                    //auto surfaceCapabilities = potential_device.getSurfaceCapabilitiesKHR(*surface);
                    auto swapchainFormats = potentialDevice.getSurfaceFormatsKHR(*surface);
                    auto swapchainPresentModes = potentialDevice.getSurfacePresentModesKHR(*surface);
                    if (swapchainFormats.empty() || swapchainPresentModes.empty())
                        return false;

                    return true;
                }
        );
        fprintf(stdout, "Selected Vulkan device %s\n", physicalDevice.getProperties().deviceName.data());

        // Create logical device and queues.
        queueFamilies = VulkanQueueFamilies::getForDevice(physicalDevice, surface).value();
        const float queuePriority = 1.0f;
        auto families = queueFamilies.uniqueFamilies();
        auto queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>();
        for (uint32_t queueFamily : families) {
            queueCreateInfos.emplace_back(
                    vk::DeviceQueueCreateFlags(),
                    queueFamily,
                    1,
                    &queuePriority

            );
        }

        auto requestedDeviceFeatures = vk::PhysicalDeviceFeatures(); // Request no features

        auto logicalDeviceCreateInfo = vk::DeviceCreateInfo();
        logicalDeviceCreateInfo.pEnabledFeatures = &requestedDeviceFeatures;
        logicalDeviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        logicalDeviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        // This is not needed but nice for legacy implementations
        logicalDeviceCreateInfo.ppEnabledLayerNames = layerNames.data();
        logicalDeviceCreateInfo.enabledLayerCount = layerNames.size();
        // Device-specific Vulkan extensions
        logicalDeviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();
        logicalDeviceCreateInfo.enabledExtensionCount = requiredDeviceExtensions.size();

        device = physicalDevice.createDeviceUnique(logicalDeviceCreateInfo);

        graphicsQueue = device->getQueue(queueFamilies.graphicsFamily, 0);
        presentQueue = device->getQueue(queueFamilies.presentFamily, 0);
    }

    // Get possible surface formats

    // Try to select a specific format first,
    // as in the tutorial https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
    surfaceFormat = selectOrFallback<vk::SurfaceFormatKHR>(
            physicalDevice.getSurfaceFormatsKHR(*surface),
            {
                    vk::SurfaceFormatKHR(
                            vk::Format::eB8G8R8A8Srgb,
                            vk::ColorSpaceKHR::eSrgbNonlinear
                    )
            }
    );

    presentMode = selectIfPossible<vk::PresentModeKHR>(
            physicalDevice.getSurfacePresentModesKHR(*surface),
            {
                    vk::PresentModeKHR::eMailbox,
                    vk::PresentModeKHR::eFifo
            }
    );

}

VulkanSetup::~VulkanSetup() {
    SDL_DestroyWindow(window);
    SDL_Quit();
}