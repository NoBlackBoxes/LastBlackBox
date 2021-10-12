#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/platform_device.h>
#include <sound/simple_card.h>
#include <linux/delay.h>

static struct asoc_simple_card_info card_info;
static struct platform_device card_device;

/*
 * Dummy callback for release
 */
void device_release_callback(struct device *dev) { /*  do nothing */ };

/*
 * Setup the card info
 */
static struct asoc_simple_card_info default_card_info = {
  .card = "NB3_ear_card",       // -> snd_soc_card.name
  .name = "simple-card_codec_link", // -> snd_soc_dai_link.name
  .codec = "snd-soc-dummy",         // "dmic-codec", // -> snd_soc_dai_link.codec_name
  .platform = "not-set.i2s",
  .daifmt = SND_SOC_DAIFMT_I2S | SND_SOC_DAIFMT_NB_NF | SND_SOC_DAIFMT_CBS_CFS,
  .cpu_dai = {
    .name = "not-set.i2s",          // -> snd_soc_dai_link.cpu_dai_name
    .sysclk = 0
  },
  .codec_dai = {
    .name = "snd-soc-dummy-dai",    //"dmic-codec", // -> snd_soc_dai_link.codec_dai_name
    .sysclk = 0
  },
};

/*
 * Setup the card device
 */
static struct platform_device default_card_device = {
  .name = "asoc-simple-card",   //module alias
  .id = 0,
  .num_resources = 0,
  .dev = {
    .release = &device_release_callback,
    .platform_data = &default_card_info, // *HACK ALERT*
  },
};

/*
 * Callback for module initialization
 */
int i2s_mic_rpi_init(void)
{
  const char *dmaengine = "bcm2708-dmaengine"; //module name
  static char *card_platform;
  int ret;

  // Report
  printk(KERN_INFO "NB3_ear: Version 0.0.1");

  // Set platform
  card_platform = "fe203000.i2s";
  printk(KERN_INFO "NB3_ear: Setting platform to %s\n", card_platform);

  // Request DMA engine module
  ret = request_module(dmaengine);
  pr_alert("request module load '%s': %d\n",dmaengine, ret);

  // Update info
  card_info = default_card_info;
  card_info.platform = card_platform;
  card_info.cpu_dai.name = card_platform;

  card_device = default_card_device;
  card_device.dev.platform_data = &card_info;

  // Register the card device
  ret = platform_device_register(&card_device);
  pr_alert("register platform device '%s': %d\n",card_device.name, ret);

  return 0;
}

/*
 * Callback for module exit
 */
void i2s_mic_rpi_exit(void)
{
  platform_device_unregister(&card_device);
  pr_alert("i2s mic module unloaded\n");
}

// Plumb it up
module_init(i2s_mic_rpi_init);
module_exit(i2s_mic_rpi_exit);
MODULE_DESCRIPTION("ASoC simple-card I2S Microphone");
MODULE_AUTHOR("Carter Nelson");
MODULE_LICENSE("GPL v2");
