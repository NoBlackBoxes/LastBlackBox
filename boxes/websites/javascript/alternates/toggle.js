// === Sidebar toggle ===
const toggle_btn = document.getElementById('toggle_sidebar');
if (toggle_btn) {
    toggle_btn.addEventListener('click', () => {
        const sidebar = document.querySelector('.sidebar');
        const container = document.querySelector('.container');

        const is_hidden = sidebar.style.display === 'none';
        sidebar.style.display = is_hidden ? 'block' : 'none';
        container.classList.toggle('sidebar-hidden', !is_hidden);
    });
}
