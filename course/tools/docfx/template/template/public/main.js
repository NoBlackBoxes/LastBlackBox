/* LastBlackBox docfx site: client-side enhancements
 *
 * Injectors:
 *   - freeze-frame for lesson GIFs (hover to play)
 *   - per-course secondary logo (host institution, optional asset)
 *   - nested left sidebar (root toc + section tocs merged into a tree)
 *   - mobile TOC toggle button (opens the sidebar offcanvas)
 *   - theme toggle (Light / Dark / Auto)
 *   - right "On this page" affix from article h2/h3
 *
 * Each injector guards against double execution with a global flag.
 * docfx Modern's native renderers are not used (they did not trigger on
 * conceptual pages in our setup), so all nav surfaces are populated here.
 */

window.LBB = window.LBB || {};

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


/* Freeze-frame lesson GIFs */
lbbOnReady(function () {
  if (window.LBB.freeze) return; window.LBB.freeze = true;

  function freezeOne(img) {
    if (img.dataset.lbbFrozen) return;
    img.dataset.lbbFrozen = "1";
    var originalSrc = img.src;
    var wrapper = img.closest("a") || img.parentNode;
    if (wrapper && wrapper.classList) wrapper.classList.add("lbb-thumb");

    var preload = new Image(); preload.src = originalSrc;

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
        wrapper.classList.add("is-playing");
        img.src = img.dataset.lbbGif;
      });
      wrapper.addEventListener("mouseleave", function () {
        if (!img.dataset.lbbStill) return;
        wrapper.classList.remove("is-playing");
        img.src = img.dataset.lbbStill;
      });
    }
  }
  document.querySelectorAll('img[src$=".gif"]').forEach(freezeOne);
});


/* Brand link rewrite: the LBB icon should take students up to the top-level
   courses landing rather than back to the current course's intro. */
lbbOnReady(function () {
  if (window.LBB.brandHref) return; window.LBB.brandHref = true;
  var brand = document.querySelector(".navbar-brand");
  if (!brand) return;
  var rel = lbbRel();
  // From a course-root page (rel = "") the landing is "../"; from any deeper
  // page (rel = "../../...") it is rel + "../".
  brand.href = (rel + "../").replace(/\/+/g, "/");
});

/* Secondary logo (host institution); convention:
     template/public/secondary_logo.{png,svg}   (the image)
     template/public/site.json                  (optional metadata)
   site.json may include `secondary_logo_url` and `secondary_logo_label`. If
   present, the logo is wrapped in an <a> pointing at that URL (new tab).
*/
lbbOnReady(function () {
  if (window.LBB.secondaryLogo) return; window.LBB.secondaryLogo = true;
  var brand = document.querySelector(".navbar-brand");
  if (!brand) return;
  if (brand.parentNode.querySelector(".lbb-secondary-logo")) return;
  var rel = lbbRel();
  var exts = ["png", "svg"];

  function placeLogo(imgUrl, metaUrl, label) {
    if (brand.parentNode.querySelector(".lbb-secondary-logo")) return;
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
    brand.parentNode.insertBefore(node, brand.nextSibling);
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


/* Nested left sidebar. Fetch root toc.json; for any entry whose href ends
   in "/" (i.e., is a section), also fetch that section's toc.json and nest
   the items under it. Render as a tree into <nav id="toc">. */
lbbOnReady(function () {
  if (window.LBB.sideNav) return; window.LBB.sideNav = true;
  var toc = document.getElementById("toc");
  if (!toc) return;
  if (toc.children.length > 0) return;
  var rel = lbbRel();
  var here = window.location.pathname;

  function htmlPath(href) { return href.replace(/\.md$/, ".html"); }

  function isCurrentItem(absHref) {
    if (!absHref) return false;
    // Compare path suffix to handle relative resolution.
    try {
      var u = new URL(absHref, window.location.href);
      return u.pathname === here ||
             (u.pathname.endsWith("/index.html") && here.endsWith(u.pathname));
    } catch (e) { return false; }
  }

  function makeLeaf(item, sectionUrlBase) {
    var li = document.createElement("li");
    li.className = "lbb-toc-leaf";
    var a = document.createElement("a");
    a.href = sectionUrlBase + htmlPath(item.href);
    a.textContent = item.name;
    if (isCurrentItem(a.href)) a.classList.add("is-active");
    li.appendChild(a);
    return li;
  }

  // Derive the section folder (e.g. "sessions") from a docfx toc item.
  // Items that point at a sub-toc have item.tocHref like "sessions/toc.html".
  function sectionDirOf(item) {
    if (item.tocHref) return item.tocHref.replace(/\/toc\.html?$/, "");
    if (item.homepage) {
      var p = item.homepage.replace(/\/?index\.html?$/, "");
      return p || null;
    }
    if (item.href) {
      if (item.href.endsWith("/")) return item.href.slice(0, -1);
      // "sessions/index.html" -> "sessions"
      var m = item.href.match(/^(.+)\/index\.html?$/);
      if (m) return m[1];
    }
    return null;
  }

  function makeSection(item, items, sectionDir) {
    var li = document.createElement("li");
    li.className = "lbb-toc-section";

    var header = document.createElement("div");
    header.className = "lbb-toc-section-header";
    var label;
    var homepageUrl = null;
    if (item.homepage) homepageUrl = rel + htmlPath(item.homepage);
    else if (item.href) homepageUrl = rel + (item.href.endsWith("/") ? item.href + "index.html" : htmlPath(item.href));
    if (homepageUrl) {
      label = document.createElement("a");
      label.href = homepageUrl;
      if (isCurrentItem(label.href)) label.classList.add("is-active");
    } else {
      label = document.createElement("span");
    }
    label.textContent = item.name;
    header.appendChild(label);
    li.appendChild(header);

    if (items && items.length) {
      var ul = document.createElement("ul");
      ul.className = "lbb-toc-sub";
      var base = rel + sectionDir + "/";
      items.forEach(function (sub) {
        if (!sub.href) return;
        var subLi = makeLeaf(sub, base);
        if (subLi.querySelector("a.is-active")) {
          li.classList.add("is-open");
          label.classList.add("is-in-active-section");
        }
        ul.appendChild(subLi);
      });
      li.appendChild(ul);
    }
    return li;
  }

  fetch(rel + "toc.json").then(function (r) {
    return r.ok ? r.json() : null;
  }).then(function (root) {
    if (!root || !root.items) return;

    // Fetch each section's sub-toc (in parallel). A section is any item
    // whose tocHref / homepage / trailing-slash href identifies a folder
    // with its own toc.json.
    var promises = root.items.map(function (item) {
      var dir = sectionDirOf(item);
      if (dir) {
        var url = (rel + dir + "/toc.json").replace(/\/+/g, "/");
        return fetch(url).then(function (r) { return r.ok ? r.json() : null; })
                         .then(function (sub) { return { item: item, sub: sub, dir: dir }; })
                         .catch(function () { return { item: item, sub: null, dir: dir }; });
      }
      return Promise.resolve({ item: item, sub: null, dir: null });
    });

    Promise.all(promises).then(function (results) {
      if (toc.children.length > 0) return;
      var ul = document.createElement("ul");
      ul.className = "lbb-toc-root";
      results.forEach(function (r) {
        if (r.sub && r.sub.items && r.dir) {
          ul.appendChild(makeSection(r.item, r.sub.items, r.dir));
        } else if (r.item.href) {
          ul.appendChild(makeLeaf(r.item, rel));
        }
      });
      toc.appendChild(ul);
    });
  }).catch(function () { /* fail soft */ });
});


/* Mobile-only TOC toggle. Adds a button to the navbar that opens the
   sidebar offcanvas. Only visible below the md breakpoint via CSS. */
lbbOnReady(function () {
  if (window.LBB.mobileTocToggle) return; window.LBB.mobileTocToggle = true;
  var brand = document.querySelector(".navbar-brand");
  if (!brand) return;
  var offcanvasTarget = document.getElementById("tocOffcanvas");
  if (!offcanvasTarget) return;

  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn border-0 d-md-none lbb-toc-toggle";
  btn.setAttribute("data-bs-toggle", "offcanvas");
  btn.setAttribute("data-bs-target", "#tocOffcanvas");
  btn.setAttribute("aria-controls", "tocOffcanvas");
  btn.setAttribute("aria-label", "Open table of contents");
  btn.innerHTML = '<i class="bi bi-list"></i>';

  // Insert at the very beginning of the navbar row (before the brand).
  brand.parentNode.insertBefore(btn, brand);
});


/* Theme toggle: cycles Light / Dark / Auto via Bootstrap data-bs-theme. */
lbbOnReady(function () {
  if (window.LBB.themeToggle) return; window.LBB.themeToggle = true;
  var navbar = document.getElementById("navbar");
  if (!navbar) return;
  if (navbar.querySelector(".lbb-theme-toggle")) return;

  var STORAGE_KEY = "lbb-theme";
  function current() { return localStorage.getItem(STORAGE_KEY) || "auto"; }
  function apply(t) {
    var resolved = t === "auto"
      ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
      : t;
    document.documentElement.setAttribute("data-bs-theme", resolved);
    btn.setAttribute("aria-label", "Theme: " + t);
    icon.className = "bi " + (t === "dark" ? "bi-moon" : t === "light" ? "bi-sun" : "bi-circle-half");
  }
  function next(t) { return t === "auto" ? "light" : t === "light" ? "dark" : "auto"; }

  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn border-0 lbb-theme-toggle";
  var icon = document.createElement("i");
  btn.appendChild(icon);

  btn.addEventListener("click", function () {
    var t = next(current());
    localStorage.setItem(STORAGE_KEY, t);
    apply(t);
  });

  // Place to the left of the search.
  var search = navbar.querySelector("form.search");
  if (search) navbar.insertBefore(btn, search);
  else navbar.appendChild(btn);

  apply(current());
});


/* Right "On this page" affix from article h2/h3 */
lbbOnReady(function () {
  if (window.LBB.affix) return; window.LBB.affix = true;
  var nav = document.getElementById("affix");
  if (!nav) return;
  if (nav.children.length > 0) return;
  var article = document.querySelector("article");
  if (!article) return;
  var headings = article.querySelectorAll("h2[id], h3[id]");
  if (headings.length === 0) return;

  var title = document.createElement("h5");
  title.textContent = "On this page";
  nav.appendChild(title);
  var ul = document.createElement("ul");
  headings.forEach(function (h) {
    var li = document.createElement("li");
    li.className = "lbb-" + h.tagName.toLowerCase();
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    ul.appendChild(li);
  });
  nav.appendChild(ul);

  var links = Array.from(nav.querySelectorAll("a"));
  var headingArr = Array.from(headings);
  function onScroll() {
    var top = window.scrollY + 120;
    var active = headingArr[0];
    for (var i = 0; i < headingArr.length; i++) {
      if (headingArr[i].offsetTop <= top) active = headingArr[i];
      else break;
    }
    links.forEach(function (l) {
      l.classList.toggle("active", l.getAttribute("href") === "#" + active.id);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
});
