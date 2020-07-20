################################################################################
#
# LBB-firmware
#
################################################################################

LBB_FIRMWARE_VERSION = 5574077183389cd4c65077ba18b59144ed6ccd6d
LBB_FIRMWARE_SITE = $(call github,raspberrypi,firmware,$(LBB_FIRMWARE_VERSION))
LBB_FIRMWARE_LICENSE = BSD-3-Clause
LBB_FIRMWARE_LICENSE_FILES = boot/LICENCE.broadcom
LBB_FIRMWARE_INSTALL_IMAGES = YES

define LBB_FIRMWARE_INSTALL_DTB
	$(foreach dtb,$(wildcard $(@D)/boot/*.dtb), \
		$(INSTALL) -D -m 0644 $(dtb) $(BINARIES_DIR)/LBB-firmware/$(notdir $(dtb))
	)
endef

define LBB_FIRMWARE_INSTALL_DTB_OVERLAYS
	for ovldtb in  $(@D)/boot/overlays/*.dtbo; do \
		$(INSTALL) -D -m 0644 $${ovldtb} $(BINARIES_DIR)/LBB-firmware/overlays/$${ovldtb##*/} || exit 1; \
	done
endef

define LBB_FIRMWARE_INSTALL_BOOT
	$(INSTALL) -D -m 0644 $(@D)/boot/start4x.elf $(BINARIES_DIR)/LBB-firmware/start4x.elf
	$(INSTALL) -D -m 0644 $(@D)/boot/fixup4x.dat $(BINARIES_DIR)/LBB-firmware/fixup4x.dat
endef

define LBB_FIRMWARE_INSTALL_IMAGES_CMDS
	$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_LBB_PATH)/package/LBB-firmware/config.txt $(BINARIES_DIR)/LBB-firmware/config.txt
	$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_LBB_PATH)/package/LBB-firmware/cmdline.txt $(BINARIES_DIR)/LBB-firmware/cmdline.txt
	$(LBB_FIRMWARE_INSTALL_BOOT)
	$(LBB_FIRMWARE_INSTALL_DTB)
	$(LBB_FIRMWARE_INSTALL_DTB_OVERLAYS)
endef

$(eval $(generic-package))
