/* Last Black Box Docfx site: client-side enhancements
 *
 * Enhancement:
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

// Keep new visitors in the light theme while preserving any saved Docfx theme
// preference in localStorage.
export default { defaultTheme: "light" };
