/**
 * @file preview.c
 * @brief Simple camera capture and display example
 * 
 * Capture frames from a Raspberry Pi camera and display them on the primary monitor
 * @author Adam Kampff
 */

// Include standard libraries
#define _GNU_SOURCE
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

// Include VK definitions
#include "vkdefs.h"

// Include VK libraries
#include "parseVK.h"
#include "mathVK.h"
#include "utilsVK.h"
#include "visionVK.h"
#include "overlayVK.h"

// Include VK modules
#include "vkDisplay.h"
#include "vkPreview.h"
#include "vkCamera.h"
#include "vkImage.h"

// Main function
int main(int argc, char *argv[])
{
    // Start Log
    setlogmask(LOG_UPTO(LOG_DEBUG));
    openlog("LBB_preview", LOG_PERROR | LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL2);

    // Initialize display
    VKDISPLAY display;
    vkDisplay_Load("config/hd_drm.vkdisplay", &display);
    vkDisplay_Initialize(&display);

    // Intialize preview renderer (OpenGL ES)
    VKPREVIEW preview;
    vkPreview_Load("config/opengl.vkpreview", &preview);
    preview.width = display.width;
    preview.height = display.height;
    vkPreview_Initialize(&preview);

    // Initialize camera
    VKCAMERA camera;
    vkCamera_Load("config/rpiv2.vkcamera", &camera);
    vkCamera_LogResources(&camera);
    vkCamera_Initialize(&camera);

    // Start Capturing
    vkCamera_Start(&camera);

    // Start profiling
    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME, &start);
    uint32_t num_frames = 600;
    for (uint32_t f = 0; f < num_frames; f++)
    {
        // Acquire next frame
        vkCamera_GrabFrame(&camera);

        // Convert to RGBA
        //vkImage_YUYVtoRGBA(&camera.frame, &preview.image);
        memcpy(preview.image.data, camera.frame.data, camera.frame.size);

        // Update Preview
        vkPreview_Update(&preview);

        // Refresh Display
        vkDisplay_Refresh(&display);

        // Report
        LOG("Got frame %d\n", f);
    }

    // End profiling
    clock_gettime(CLOCK_REALTIME, &finish);
    long seconds = finish.tv_sec - start.tv_sec;
    long ns = finish.tv_nsec - start.tv_nsec;
    if (start.tv_nsec > finish.tv_nsec)
    {
        --seconds;
        ns += 1000000000;
    }
    float elapsed = (float)seconds + (float)ns / 1000000000.f;
    float fps = (float)num_frames / elapsed;
    LOG("Time elpased: %ld s, %ld ns", seconds, ns);
    LOG("Time elpased: %f", elapsed);
    LOG("FPS: %f", fps);

    // Stop Capturing
    vkCamera_Stop(&camera);
    LOG("Camera Stopped.");

    // Cleanup
    vkCamera_Cleanup(&camera);
    vkPreview_Cleanup(&preview);
    vkDisplay_Cleanup(&display);
    closelog();
}
// FIN