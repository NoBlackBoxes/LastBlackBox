/* Last Black Box Docfx site: client-side enhancements
 *
 * Enhancements:
 *   - freeze-frame for lesson GIFs (hover to play)
 *   - per-course secondary logo (host institution, optional asset)
 *   - global brand link back to the courses landing page
 *
 * Docfx imports this file as its customization module and owns search,
 * breadcrumbs, the page outline, and the theme picker.
 */

/* ===========================================================================
   Shared helpers
   ========================================================================== */
function lbbOnReady(fn) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", fn);
  } else {
    fn();
  }
}

function lbbRel() {
  var m = document.querySelector('meta[name="docfx:rel"]');
  return m ? (m.getAttribute("content") || "") : "";
}


/* ===========================================================================
   Lesson GIF previews
   Capture the first frame as a still image, then restore the original GIF
   while the pointer is over its link or wrapper.
   ========================================================================== */
lbbOnReady(function () {
  function freezeOne(img) {
    if (img.dataset.lbbFrozen) return;
    img.dataset.lbbFrozen = "1";
    var originalSrc = img.src;
    var wrapper = img.closest("a") || img.parentNode;
    if (wrapper && wrapper.classList) wrapper.classList.add("lbb-thumb");

    function captureAndSwap() {
      try {
        var w = img.naturalWidth, h = img.naturalHeight;
        if (!w || !h) return;
        var canvas = document.createElement("canvas");
        canvas.width = w; canvas.height = h;
        canvas.getContext("2d").drawImage(img, 0, 0, w, h);
        var still = canvas.toDataURL("image/png");
        img.dataset.lbbStill = still;
        img.dataset.lbbGif = originalSrc;
        img.src = still;
      } catch (e) { /* tainted canvas */ }
    }
    if (img.complete && img.naturalWidth) captureAndSwap();
    else img.addEventListener("load", captureAndSwap, { once: true });

    if (wrapper) {
      wrapper.addEventListener("mouseenter", function () {
        if (!img.dataset.lbbGif) return;
        img.src = img.dataset.lbbGif;
      });
      wrapper.addEventListener("mouseleave", function () {
        if (!img.dataset.lbbStill) return;
        img.src = img.dataset.lbbStill;
      });
    }
  }
  document.querySelectorAll('img[src$=".gif"]').forEach(freezeOne);
});


/* ===========================================================================
   Global brand
   Keep the navbar identity stable across course sites and link it to the
   top-level courses landing. The course name remains the page H1.
   ========================================================================== */
lbbOnReady(function () {
  var brand = document.querySelector(".navbar-brand");
  if (!brand) return;
  var rel = lbbRel();
  // From a course-root page (rel = "") the landing is "../"; from any deeper
  // page (rel = "../../...") it is rel + "../".
  brand.href = (rel + "../").replace(/\/+/g, "/");
});

/* ===========================================================================
   Optional host identity
   Shown only on the course landing page, below the course title.
   Asset convention:
     template/public/secondary_logo.{png,svg}   (the image)
     template/public/site.json                  (optional metadata)
   site.json may include `secondary_logo_url` and `secondary_logo_label`. If
   present, the logo is linked to that URL. Deeper lesson pages omit this block.
   ========================================================================== */
lbbOnReady(function () {
  var rel = lbbRel();
  if (rel) return;
  var heading = document.querySelector("article h1");
  if (!heading || document.querySelector(".lbb-partner")) return;
  var exts = ["png", "svg"];

  function placeLogo(imgUrl, metaUrl, label) {
    if (document.querySelector(".lbb-partner")) return;
    var partner = document.createElement("div");
    partner.className = "lbb-partner";
    var caption = document.createElement("span");
    caption.className = "lbb-partner-label";
    caption.textContent = "In partnership with";
    partner.appendChild(caption);

    var img = document.createElement("img");
    img.src = imgUrl;
    img.alt = label || "Host institution";
    img.className = "lbb-secondary-logo";
    var node = img;
    if (metaUrl) {
      var a = document.createElement("a");
      a.href = metaUrl;
      a.target = "_blank";
      a.rel = "noopener";
      a.className = "lbb-secondary-logo-link";
      a.setAttribute("aria-label", label || "Host institution");
      a.appendChild(img);
      node = a;
    }
    partner.appendChild(node);
    heading.insertAdjacentElement("afterend", partner);
  }

  function trySiteMeta(imgUrl) {
    fetch(rel + "public/site.json").then(function (r) {
      return r.ok ? r.json() : null;
    }).then(function (cfg) {
      placeLogo(imgUrl, cfg && cfg.secondary_logo_url, cfg && cfg.secondary_logo_label);
    }).catch(function () { placeLogo(imgUrl, null, null); });
  }

  (function tryNext(i) {
    if (i >= exts.length) return;
    var url = rel + "public/secondary_logo." + exts[i];
    var probe = new Image();
    probe.onload = function () { trySiteMeta(url); };
    probe.onerror = function () { tryNext(i + 1); };
    probe.src = url;
  })(0);
});


// Keep new visitors in the light theme while preserving any saved Docfx theme
// preference in localStorage.
export default { defaultTheme: "light" };
