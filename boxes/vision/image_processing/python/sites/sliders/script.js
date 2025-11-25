/**
 * Sends a command to the server via HTTP GET request.
 * @param {string} command - The command name.
 * @param {string|number} value - The value to send.
 */
function sendCommand(command, value) {
    fetch(`/command/${command}-${value}`, { method: 'GET' })
        .then(() => console.log(`${command} = ${value} sent`))
        .catch(error => console.error('Error:', error));
}

document.addEventListener('DOMContentLoaded', () => {
  const hue_slider = document.getElementById('hue');
  const hue_value_label = document.getElementById('hue-value');
  const sat_slider = document.getElementById('sat');
  const sat_value_label = document.getElementById('sat-value');
  const val_slider = document.getElementById('val');
  const val_value_label = document.getElementById('val-value');

  // set initial display
  hue_value_label.textContent = hue_slider.value;
  sat_value_label.textContent = sat_slider.value;
  val_value_label.textContent = val_slider.value;

  // live updates as the user drags
  hue_slider.addEventListener('input', () => {
    hue_value_label.textContent = hue_slider.value;
    sendCommand('hue', hue_slider.value);
  });

  // live updates as the user drags
  sat_slider.addEventListener('input', () => {
    sat_value_label.textContent = sat_slider.value;
    sendCommand('sat', sat_slider.value);
  });

  // live updates as the user drags
  val_slider.addEventListener('input', () => {
    val_value_label.textContent = val_slider.value;
    sendCommand('value', val_slider.value);
  });
});
