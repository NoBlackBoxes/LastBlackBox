// SPDX-License-Identifier: GPL-2.0-only
/* Copyright (c) 2023 - Voight-Kampff
 *
 * nb3_audio.c -- NB3_AUDIO ALSA SoC Codec driver (based on max98357a.c and ics4342.c)
 */

#include <linux/acpi.h>
#include <linux/delay.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/gpio.h>
#include <linux/gpio/consumer.h>
#include <linux/kernel.h>
#include <linux/mod_devicetable.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <sound/pcm.h>
#include <sound/soc.h>
#include <sound/soc-dai.h>
#include <sound/soc-dapm.h>

struct nb3_audio_priv {
	struct gpio_desc *sdmode;
	unsigned int sdmode_delay;
	int sdmode_switch;
};

static int nb3_audio_daiops_trigger(struct snd_pcm_substream *substream,
		int cmd, struct snd_soc_dai *dai)
{
	struct snd_soc_component *component = dai->component;
	struct nb3_audio_priv *nb3_audio =
		snd_soc_component_get_drvdata(component);

	if (!nb3_audio->sdmode)
		return 0;

	switch (cmd) {
	case SNDRV_PCM_TRIGGER_START:
	case SNDRV_PCM_TRIGGER_RESUME:
	case SNDRV_PCM_TRIGGER_PAUSE_RELEASE:
		mdelay(nb3_audio->sdmode_delay);
		if (nb3_audio->sdmode_switch) {
			gpiod_set_value(nb3_audio->sdmode, 1);
			dev_dbg(component->dev, "set sdmode to 1");
		}
		break;
	case SNDRV_PCM_TRIGGER_STOP:
	case SNDRV_PCM_TRIGGER_SUSPEND:
	case SNDRV_PCM_TRIGGER_PAUSE_PUSH:
		gpiod_set_value(nb3_audio->sdmode, 0);
		dev_dbg(component->dev, "set sdmode to 0");
		break;
	}

	return 0;
}

static int nb3_audio_sdmode_event(struct snd_soc_dapm_widget *w,
		struct snd_kcontrol *kcontrol, int event)
{
	struct snd_soc_component *component =
		snd_soc_dapm_to_component(w->dapm);
	struct nb3_audio_priv *nb3_audio =
		snd_soc_component_get_drvdata(component);

	if (event & SND_SOC_DAPM_POST_PMU)
		nb3_audio->sdmode_switch = 1;
	else if (event & SND_SOC_DAPM_POST_PMD)
		nb3_audio->sdmode_switch = 0;

	return 0;
}

static const struct snd_soc_dapm_widget nb3_audio_dapm_widgets[] = {
	SND_SOC_DAPM_OUTPUT("Speaker"),
	SND_SOC_DAPM_OUT_DRV_E("SD_MODE", SND_SOC_NOPM, 0, 0, NULL, 0,
			nb3_audio_sdmode_event,
			SND_SOC_DAPM_POST_PMU | SND_SOC_DAPM_POST_PMD),
};

static const struct snd_soc_dapm_route nb3_audio_dapm_routes[] = {
	{"SD_MODE", NULL, "HiFi Playback"},
	{"Speaker", NULL, "SD_MODE"},
};

static const struct snd_soc_component_driver nb3_audio_component_driver = {
	.dapm_widgets		= nb3_audio_dapm_widgets,
	.num_dapm_widgets	= ARRAY_SIZE(nb3_audio_dapm_widgets),
	.dapm_routes		= nb3_audio_dapm_routes,
	.num_dapm_routes	= ARRAY_SIZE(nb3_audio_dapm_routes),
	.idle_bias_on		= 1,
	.use_pmdown_time	= 1,
	.endianness		= 1,
};

static const struct snd_soc_dai_ops nb3_audio_dai_ops = {
	.trigger        = nb3_audio_daiops_trigger,
};

static struct snd_soc_dai_driver nb3_audio_dai_driver = {
	.name = "HiFi",
	.playback = {
		.stream_name	= "HiFi Playback",
		.formats	= SNDRV_PCM_FMTBIT_S16 |
					SNDRV_PCM_FMTBIT_S24 |
					SNDRV_PCM_FMTBIT_S32,
		.rates		= SNDRV_PCM_RATE_8000 |
					SNDRV_PCM_RATE_16000 |
					SNDRV_PCM_RATE_32000 |
					SNDRV_PCM_RATE_44100 |
					SNDRV_PCM_RATE_48000 |
					SNDRV_PCM_RATE_88200 |
					SNDRV_PCM_RATE_96000,
		.rate_min	= 8000,
		.rate_max	= 96000,
		.channels_min	= 1,
		.channels_max	= 2,
	},
	.capture = {
        .stream_name    = "HiFi Capture",
        .formats        = SNDRV_PCM_FMTBIT_S24_LE |
                        SNDRV_PCM_FMTBIT_S32,
        .rates          = SNDRV_PCM_RATE_8000 |
                        SNDRV_PCM_RATE_16000 |
                        SNDRV_PCM_RATE_32000 |
                        SNDRV_PCM_RATE_44100 |
                        SNDRV_PCM_RATE_48000,
        .rate_min       = 7190,
        .rate_max       = 52800,
        .channels_min   = 1,
        .channels_max   = 2,
    },
	.ops    = &nb3_audio_dai_ops,
};

static int nb3_audio_platform_probe(struct platform_device *pdev)
{
	struct nb3_audio_priv *nb3_audio;
	int ret;

	nb3_audio = devm_kzalloc(&pdev->dev, sizeof(*nb3_audio), GFP_KERNEL);
	if (!nb3_audio)
		return -ENOMEM;

	nb3_audio->sdmode = devm_gpiod_get_optional(&pdev->dev,
				"sdmode", GPIOD_OUT_LOW);
	if (IS_ERR(nb3_audio->sdmode))
		return PTR_ERR(nb3_audio->sdmode);

	ret = device_property_read_u32(&pdev->dev, "sdmode-delay",
					&nb3_audio->sdmode_delay);
	if (ret) {
		nb3_audio->sdmode_delay = 0;
		dev_dbg(&pdev->dev,
			"no optional property 'sdmode-delay' found, "
			"default: no delay\n");
	}

	dev_set_drvdata(&pdev->dev, nb3_audio);

	return devm_snd_soc_register_component(&pdev->dev,
			&nb3_audio_component_driver,
			&nb3_audio_dai_driver, 1);
}

#ifdef CONFIG_OF
static const struct of_device_id nb3_audio_device_id[] = {
	{ .compatible = "maxim,max98357a" },
	{}
};
MODULE_DEVICE_TABLE(of, nb3_audio_device_id);
#endif

#ifdef CONFIG_ACPI
static const struct acpi_device_id nb3_audio_acpi_match[] = {
	{ "MX98357A", 0 },
	{},
};
MODULE_DEVICE_TABLE(acpi, nb3_audio_acpi_match);
#endif

static struct platform_driver nb3_audio_platform_driver = {
	.driver = {
		.name = "nb3_audio",
		.of_match_table = of_match_ptr(nb3_audio_device_id),
		.acpi_match_table = ACPI_PTR(nb3_audio_acpi_match),
	},
	.probe	= nb3_audio_platform_probe,
};
module_platform_driver(nb3_audio_platform_driver);

MODULE_DESCRIPTION("NB3 Audio (Ear and Mouth) Driver");
MODULE_LICENSE("GPL v2");
