//
// Created by samuel on 22/08/2020.
//
#include "VulkanWindow.h"

#include <SDL_vulkan.h>
#include "util/fatal_error.h"

#if !NDEBUG
VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData){

    auto severity = vk::to_string(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity));
    auto type = vk::to_string(vk::DebugUtilsMessageTypeFlagsEXT(messageType));

    auto print_to = (messageSeverity | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? stderr : stdout;
    fprintf(print_to, "VulkanDebug [%s] [%s]: [%s] %s [%d]\n",
            type.c_str(), severity.c_str(),
            pCallbackData->pMessageIdName, pCallbackData->pMessage, pCallbackData->messageIdNumber);

    // VK_FALSE => don't stop the application
    return VK_FALSE;
}
#endif

VulkanWindow::VulkanWindow(const vk::ApplicationInfo& app_info, Size<size_t> window_size) : dispatch_loader() {
    window = SDL_CreateWindow(
            app_info.pApplicationName,
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                window_size.x, window_size.y,
                SDL_WINDOW_VULKAN
            );

    {
        uint32_t extension_count;
        check_sdl_error(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, nullptr));
        auto extension_names = std::vector<const char *>(extension_count);
        check_sdl_error(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extension_names.data()));


        auto layer_names = std::vector<const char *>();
        if (VulkanDebug) {
            layer_names.push_back("VK_LAYER_KHRONOS_validation");
            extension_names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        auto create_info = vk::InstanceCreateInfo(
                vk::InstanceCreateFlags(),
                &app_info,
                layer_names.size(),
                layer_names.data(),
                extension_names.size(),
                extension_names.data());
        instance = vk::createInstanceUnique(create_info);

        dispatch_loader.init(*instance, vkGetInstanceProcAddr);

        if (VulkanDebug) {
            auto messenger_create = vk::DebugUtilsMessengerCreateInfoEXT(
                    vk::DebugUtilsMessengerCreateFlagsEXT(),
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                    &vulkanDebug,
                    nullptr
            );

            debug_messenger = instance->createDebugUtilsMessengerEXTUnique(
                    messenger_create,
                    nullptr,
                    dispatch_loader
                    );
        }
    }

    {
        VkSurfaceKHR c_surface = nullptr;
        check_sdl_error(SDL_Vulkan_CreateSurface(window, *instance, &c_surface));
        surface = vk::UniqueSurfaceKHR(c_surface, *instance);
    }
}
VulkanWindow::~VulkanWindow() {
    SDL_DestroyWindow(window);
    SDL_Quit();
}
void VulkanWindow::main_loop() {
}
void VulkanWindow::check_sdl_error(SDL_bool success) {
    FATAL_ERROR_IF(!success, "SDL Error: %s\n", SDL_GetError());
}
void VulkanWindow::check_vulkan_error(vk::Result result) {
    FATAL_ERROR_IF(result != vk::Result::eSuccess, "Vulkan Error: %s\n", vk::to_string(result).c_str());
}
