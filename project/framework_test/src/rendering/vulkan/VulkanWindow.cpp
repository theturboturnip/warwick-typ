//
// Created by samuel on 22/08/2020.
//
#include "VulkanWindow.h"

#include "VulkanQueueFamilies.h"
#include "VulkanRenderPass.h"
#include "util/fatal_error.h"
#include <SDL_vulkan.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

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

template<typename DeviceSelectorType>
vk::PhysicalDevice selectDevice(const vk::UniqueInstance& instance, DeviceSelectorType selector) {
    auto devices = instance->enumeratePhysicalDevices();

    for (const auto& device : devices) {
        if (selector(device))
            return device;
    }
    FATAL_ERROR("Could not find a suitable device.\n");
}

VulkanWindow::VulkanWindow(const vk::ApplicationInfo& app_info, Size<size_t> window_size) : window_size(window_size), dispatch_loader() {
    window = SDL_CreateWindow(
            app_info.pApplicationName,
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                window_size.x, window_size.y,
                SDL_WINDOW_VULKAN
            );


    uint32_t extension_count;
    check_sdl_error(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, nullptr));
    auto extension_names = std::vector<const char *>(extension_count);
    check_sdl_error(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extension_names.data()));


    auto layer_names = std::vector<const char *>();
    if (VulkanDebug) {
        layer_names.push_back("VK_LAYER_KHRONOS_validation");
        extension_names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    {
        printf("Creating Vulkan instance\nLayers\n");
        for (const auto layer_name : layer_names) {
            printf("\t%s\n", layer_name);
        }
        printf("Extensions\n");
        for (const auto extension_name : extension_names) {
            printf("\t%s\n", extension_name);
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
    }

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

    {
        VkSurfaceKHR c_surface = nullptr;
        check_sdl_error(SDL_Vulkan_CreateSurface(window, *instance, &c_surface));
        surface = vk::UniqueSurfaceKHR(c_surface, *instance);
    }

    {
        std::vector<const char*> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        physicalDevice = selectDevice(instance, [this, &requiredDeviceExtensions](vk::PhysicalDevice potential_device){
            auto deviceProperties = potential_device.getProperties();
            if (deviceProperties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu)
                return false; // Only accept discrete GPUs

            auto potential_queueFamilies = VulkanQueueFamilies::fill_from_vulkan(potential_device, surface);
            if (!potential_queueFamilies.complete())
                return false; // Can't support all of the queues we want

            auto availableExtensions = potential_device.enumerateDeviceExtensionProperties();
            // TODO - why does this work? Is there an implicit conversion between const char* and std::string??
            std::set<std::string> requiredExtensionsSet(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());

            for (const auto& extension : availableExtensions) {
                requiredExtensionsSet.erase(std::string(extension.extensionName));
            }
            if (!requiredExtensionsSet.empty())
                return false; // Not all extensions present

            //auto surfaceCapabilities = potential_device.getSurfaceCapabilitiesKHR(*surface);
            auto swapchainFormats = potential_device.getSurfaceFormatsKHR(*surface);
            auto swapchainPresentModes = potential_device.getSurfacePresentModesKHR(*surface);
            if (swapchainFormats.empty() || swapchainPresentModes.empty())
                return false;

            return true;
        });
        queueFamilies = VulkanQueueFamilies::fill_from_vulkan(physicalDevice, surface);
        fprintf(stdout, "Selected Vulkan device %s\n", physicalDevice.getProperties().deviceName.data());

        const float queuePriority = 1.0f;
        auto families = queueFamilies.get_families();
        auto queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>();
        for (uint32_t queueFamily : families) {
            queueCreateInfos.push_back(
                    vk::DeviceQueueCreateInfo(
                            vk::DeviceQueueCreateFlags(),
                            queueFamily,
                            1,
                            &queuePriority
                    )
                );
        }

        auto requestedDeviceFeatures = vk::PhysicalDeviceFeatures();

        auto logicalDeviceCreateInfo = vk::DeviceCreateInfo();
        logicalDeviceCreateInfo.pEnabledFeatures = &requestedDeviceFeatures;
        logicalDeviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        logicalDeviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        // This is not needed but nice for legacy implementations
        logicalDeviceCreateInfo.ppEnabledLayerNames = layer_names.data();
        logicalDeviceCreateInfo.enabledLayerCount = layer_names.size();
        // Device-specific Vulkan extensions
        logicalDeviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();
        logicalDeviceCreateInfo.enabledExtensionCount = requiredDeviceExtensions.size();

        logicalDevice = physicalDevice.createDeviceUnique(logicalDeviceCreateInfo);

        graphicsQueue = logicalDevice->getQueue(queueFamilies.graphics_family.value(), 0);
        presentQueue = logicalDevice->getQueue(queueFamilies.present_family.value(), 0);
    }

    {
        // The tutorial https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
        // tries to select a certain format first, but we don't care about color spaces or exact formats rn.
        swapchainProps.surfaceFormat = physicalDevice.getSurfaceFormatsKHR(*surface)[0];

        auto swapchainPresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
        if (std::find(swapchainPresentModes.begin(), swapchainPresentModes.end(), vk::PresentModeKHR::eMailbox) != swapchainPresentModes.end())
            swapchainProps.presentMode = vk::PresentModeKHR::eMailbox;
        else
            swapchainProps.presentMode = vk::PresentModeKHR::eFifo; // This is mandated to always be present

        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            // If the surface currently has an extent, just use that for the swapchain
            swapchainProps.extents = surfaceCapabilities.currentExtent;
        } else {
            // The surface doesn't specify an extent to use, so select the one we want.
            // The tutorial just clamps the x/y inside the minimum/maximum ranges. If this ever happens everything is going to look weird, so we just stop.
            if (window_size.x < surfaceCapabilities.minImageExtent.width || surfaceCapabilities.maxImageExtent.width < window_size.x) {
                FATAL_ERROR("Window width %zu out of range [%u, %u]\n", window_size.x, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
            }
            if (window_size.y < surfaceCapabilities.minImageExtent.height || surfaceCapabilities.maxImageExtent.height < window_size.y) {
                FATAL_ERROR("Window height %zu out of range [%u, %u]\n", window_size.y, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
            }

            swapchainProps.extents = vk::Extent2D(window_size.x, window_size.y);
        }

        // If we just took the minimum, we could end up having to wait on the driver before getting another image.
        // Get 1 extra, so 1 is always free at any given time
        swapchainProps.imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 &&
            swapchainProps.imageCount > surfaceCapabilities.maxImageCount) {
            // Make sure we don't exceed the maximum
            swapchainProps.imageCount = surfaceCapabilities.maxImageCount;
        }

        auto swapchainCreateInfo = vk::SwapchainCreateInfoKHR();
        swapchainCreateInfo.surface = *surface;

        swapchainCreateInfo.presentMode = swapchainProps.presentMode;
        swapchainCreateInfo.minImageCount = swapchainProps.imageCount;
        swapchainCreateInfo.imageExtent = swapchainProps.extents;
        swapchainCreateInfo.imageFormat = swapchainProps.surfaceFormat.format;
        swapchainCreateInfo.imageColorSpace = swapchainProps.surfaceFormat.colorSpace;
        swapchainCreateInfo.imageArrayLayers = 1; // We're not rendering in stereoscopic 3D => set this to 1
        swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // Use eColorAttachment so we can directly render to the swapchain images.

        auto queueFamilyVector = std::vector<uint32_t>({queueFamilies.graphics_family.value(), queueFamilies.present_family.value()});
        if (queueFamilies.graphics_family != queueFamilies.present_family) {
            // The swapchain images need to be able to be used by both families.
            // Use Concurrent mode to make that possible.
            swapchainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            swapchainCreateInfo.queueFamilyIndexCount = queueFamilyVector.size();
            swapchainCreateInfo.pQueueFamilyIndices = queueFamilyVector.data();
        } else {
            // Same queue families => images can be owned exclusively by that queue family.
            // In this case we don't need to specify the different queues, because there is only one.
            swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
            swapchainCreateInfo.queueFamilyIndexCount = 0;
            swapchainCreateInfo.pQueueFamilyIndices = nullptr;
        }

        // Extra stuff
        // Set the swapchain rotation to the current rotation of the surface
        swapchainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
        // Don't apply the alpha channel as transparency to the window
        // i.e. if a pixel has alpha = 0 in the presented image it will be opqaue in the window system
        swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        swapchainCreateInfo.clipped = VK_TRUE;
        swapchainCreateInfo.oldSwapchain = nullptr;

        swapchain = logicalDevice->createSwapchainKHRUnique(swapchainCreateInfo);
        swapchainImages = logicalDevice->getSwapchainImagesKHR(*swapchain);
        swapchainImageViews.clear();
        for (vk::Image image : swapchainImages) {
            swapchainImageViews.push_back(make_identity_view(image, swapchainProps.surfaceFormat.format));
        }
    }

    {
        renderPass = VulkanRenderPass(*logicalDevice, swapchainProps.surfaceFormat.format, VulkanRenderPass::Position::PipelineStartAndEnd);//
    }

    {
        pipelines = std::make_unique<VulkanPipelineSet>(*logicalDevice, *renderPass, window_size);
    }

    {
        swapchainFramebuffers.clear();
        for (const auto& imageView : swapchainImageViews) {
            auto framebufferCreateInfo = vk::FramebufferCreateInfo();
            framebufferCreateInfo.renderPass = *renderPass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = &(*imageView);
            framebufferCreateInfo.width = window_size.x;
            framebufferCreateInfo.height = window_size.y;
            framebufferCreateInfo.layers = 1;

            swapchainFramebuffers.push_back(logicalDevice->createFramebufferUnique(framebufferCreateInfo));
        }
    }

    {
        // TODO - Make init order consistent with declare order
        //  objects are declared in VulkanWindow.h in a specific order so parents are destroyed after children
        //  i.e. command buffers are destroyed *before* the pool is destroyed
        //  so the pool is declared before it's children.
        auto poolInfo = vk::CommandPoolCreateInfo();
        poolInfo.queueFamilyIndex = queueFamilies.graphics_family.value();
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer; // Allow command buffers to be reset outside of the pool?

        cmdPool = logicalDevice->createCommandPoolUnique(poolInfo);

        perFrameCommandBuffers.clear();
        auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
        cmdBufferAlloc.commandPool = *cmdPool;
        cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
        cmdBufferAlloc.commandBufferCount = swapchainProps.imageCount;
        perFrameCommandBuffers = logicalDevice->allocateCommandBuffersUnique(cmdBufferAlloc);

        cmdBufferAlloc.commandBufferCount = 1;
        //imguiCmdBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(cmdBufferAlloc)[0]);

        // Record command buffers
        // TODO - Turn struct-of-arrays for frame data into array-of-structs, then this i isn't needed
//        for (size_t i = 0; i < perFrameCommandBuffers.size(); i++) {
//            const auto& cmdBuffer = *perFrameCommandBuffers[i];
//
//            auto beginInfo = vk::CommandBufferBeginInfo();
//            auto renderPassInfo = vk::RenderPassBeginInfo();
//            renderPassInfo.renderPass = *renderPass;
//            renderPassInfo.framebuffer = *swapchainFramebuffers[i];
//            renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
//            renderPassInfo.renderArea.extent = vk::Extent2D{(uint32_t)window_size.x, (uint32_t)window_size.y};
//            auto clearColor = vk::ClearValue(vk::ClearColorValue());
//            clearColor.color.setFloat32({1.0f, 0.0f, 1.0f, 1.0f});
//            renderPassInfo.clearValueCount = 1;
//            renderPassInfo.pClearValues = &clearColor;
//
//            cmdBuffer.begin(beginInfo);
//            cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
//            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines->redTriangle);
//            cmdBuffer.draw(3, 1, 0, 0);
//            cmdBuffer.endRenderPass();
//            cmdBuffer.end();
//        }
    }
    
    {
        std::vector<vk::DescriptorPoolSize> pool_sizes =
                {
                        { vk::DescriptorType::eSampler, 1000 },
                        { vk::DescriptorType::eCombinedImageSampler, 1000 },
                        { vk::DescriptorType::eSampledImage, 1000 },
                        { vk::DescriptorType::eStorageImage, 1000 },
                        { vk::DescriptorType::eUniformTexelBuffer, 1000 },
                        { vk::DescriptorType::eStorageTexelBuffer, 1000 },
                        { vk::DescriptorType::eUniformBuffer, 1000 },
                        { vk::DescriptorType::eStorageBuffer, 1000 },
                        { vk::DescriptorType::eUniformBufferDynamic, 1000 },
                        { vk::DescriptorType::eStorageBufferDynamic, 1000 },
                        { vk::DescriptorType::eInputAttachment, 1000 }
                };
        vk::DescriptorPoolCreateInfo pool_info = {};
        pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;// VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * pool_sizes.size();
        pool_info.poolSizeCount = pool_sizes.size();
        pool_info.pPoolSizes = pool_sizes.data();
        descriptorPool = logicalDevice->createDescriptorPoolUnique(pool_info);
    }

    {
        auto semaphoreInfo = vk::SemaphoreCreateInfo();
        hasImage = logicalDevice->createSemaphoreUnique(semaphoreInfo);
        renderFinished = logicalDevice->createSemaphoreUnique(semaphoreInfo);
    }
}
VulkanWindow::~VulkanWindow() {
    if (logicalDevice)
        logicalDevice->waitIdle();

    SDL_DestroyWindow(window);
    SDL_Quit();
}
void VulkanWindow::main_loop() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = *logicalDevice;
    init_info.QueueFamily = queueFamilies.graphics_family.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = nullptr;
    init_info.DescriptorPool = *descriptorPool;
    init_info.Allocator = nullptr;
    init_info.MinImageCount = swapchainProps.imageCount; // TODO - this isn't right
    init_info.ImageCount = swapchainProps.imageCount;
    init_info.CheckVkResultFn = nullptr; // TODO
    ImGui_ImplVulkan_Init(&init_info, *renderPass);

    {
        auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
        cmdBufferAlloc.commandPool = *cmdPool;
        cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
        cmdBufferAlloc.commandBufferCount = 1;
        auto cmdBuffers = logicalDevice->allocateCommandBuffersUnique(cmdBufferAlloc);
        const auto& cmdBuffer = *cmdBuffers[0];

        vk::CommandBufferBeginInfo begin_info = {};
        begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        {
            cmdBuffer.begin(begin_info);
            ImGui_ImplVulkan_CreateFontsTexture(cmdBuffer);
            cmdBuffer.end();
        }

        vk::SubmitInfo submitInfo = {};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;
        graphicsQueue.submit({submitInfo}, nullptr);

        logicalDevice->waitIdle();
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    bool showDemoWindow = true;

    while (true) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                goto end;
            else
                ImGui_ImplSDL2_ProcessEvent(&event);
        }

        uint32_t swFrameIndex;
        logicalDevice->acquireNextImageKHR(*swapchain, std::numeric_limits<uint64_t>::max(), *hasImage, nullptr, &swFrameIndex);

        const auto& cmdBuffer = *perFrameCommandBuffers[swFrameIndex];
        cmdBuffer.reset({});

        auto beginInfo = vk::CommandBufferBeginInfo();
        auto renderPassInfo = vk::RenderPassBeginInfo();
        renderPassInfo.renderPass = *renderPass;
        renderPassInfo.framebuffer = *swapchainFramebuffers[swFrameIndex];
        renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
        renderPassInfo.renderArea.extent = vk::Extent2D{(uint32_t)window_size.x, (uint32_t)window_size.y};
        auto clearColor = vk::ClearValue(vk::ClearColorValue());
        clearColor.color.setFloat32({1.0f, 0.0f, 1.0f, 1.0f});
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        cmdBuffer.begin(beginInfo);
        cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines->redTriangle);
        cmdBuffer.draw(3, 1, 0, 0);

        {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplSDL2_NewFrame(window);
            ImGui::NewFrame();

            if (showDemoWindow)
                ImGui::ShowDemoWindow(&showDemoWindow);

            ImGui::Render();

            ImDrawData* draw_data = ImGui::GetDrawData();
            ImGui_ImplVulkan_RenderDrawData(draw_data, cmdBuffer);
        }

        cmdBuffer.endRenderPass();
        cmdBuffer.end();

        // TODO - Dispatch the CUDA simulation with a CUDAified version of the renderFinished semaphore.
        //  Then, lock the draw behind semaphores for both 1. getting the next image 2. CUDA finishing.
        logicalDevice->waitIdle();

        vk::SubmitInfo submitInfo{};
        vk::Semaphore waitSemaphores[] = {*hasImage};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        vk::CommandBuffer cmdBuffers[] = {
            *perFrameCommandBuffers[swFrameIndex]
        };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = cmdBuffers;

        vk::Semaphore signalSemaphores[] = {*renderFinished};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        graphicsQueue.submit({submitInfo}, nullptr);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &(*swapchain);
        presentInfo.pImageIndices = &swFrameIndex;
        presentInfo.pResults = nullptr;
        presentQueue.presentKHR(presentInfo);
    }

    end:;

    logicalDevice->waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
}
void VulkanWindow::check_sdl_error(SDL_bool success) {
    FATAL_ERROR_IF(!success, "SDL Error: %s\n", SDL_GetError());
}
void VulkanWindow::check_vulkan_error(vk::Result result) {
    FATAL_ERROR_IF(result != vk::Result::eSuccess, "Vulkan Error: %s\n", vk::to_string(result).c_str());
}
vk::UniqueImageView VulkanWindow::make_identity_view(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags) {
    auto createInfo = vk::ImageViewCreateInfo();
    createInfo.image = image;
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = format;

    createInfo.components.r = vk::ComponentSwizzle::eIdentity;
    createInfo.components.g = vk::ComponentSwizzle::eIdentity;
    createInfo.components.b = vk::ComponentSwizzle::eIdentity;
    createInfo.components.a = vk::ComponentSwizzle::eIdentity;

    createInfo.subresourceRange.aspectMask = aspectFlags;
    // We don't do any mipmapping/texture arrays ever - only use the first mip level, and the first array layer
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    return logicalDevice->createImageViewUnique(createInfo);
}
