// === Sidebar toggle ===
const toggle_btn = document.getElementById('toggle_sidebar');
if (toggle_btn) {
    toggle_btn.addEventListener('click', () => {
        const sidebar = document.querySelector('.sidebar');
        const container = document.querySelector('.container');

        const is_hidden = sidebar.style.display === 'none';         // Is the sidebar hidden?
        sidebar.style.display = is_hidden ? 'block' : 'none';       // Toggle visibility
        container.classList.toggle('sidebar-hidden', !is_hidden);   // Toggle CSS class "sidebar.hidden"
    });
}
