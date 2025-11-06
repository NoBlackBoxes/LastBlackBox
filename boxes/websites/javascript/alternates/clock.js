// === Live clock ===
function start_clock(clock_el, date_el) {
    function tick() {
        const now = new Date();
        clock_el.textContent = now.toLocaleTimeString(); // handles padding and locale
        date_el.textContent = now.toLocaleDateString(undefined, {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    tick();
    setInterval(tick, 1000);
}

const clock_el = document.getElementById('clock');
const date_el  = document.getElementById('date');
if (clock_el && date_el) {
    start_clock(clock_el, date_el);
}
