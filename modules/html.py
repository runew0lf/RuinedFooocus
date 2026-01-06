css = """
.loader-container {
  display: flex; /* Use flex to align items horizontally */
  align-items: center; /* Center items vertically within the container */
  white-space: nowrap; /* Prevent line breaks within the container */
  overflow-x:hidden;
  overflow-y:scroll;
}
/* custom css start*/
#component-19 {
    position: ;
}
#component-7 {
    height: 100% !important;
}
.gradio-container.gradio-container-5-35-0 .contain .type_row{
    height: 150px;
}
/* custom css ends*/
/* Style the progress bar */
progress {
  appearance: none; /* Remove default styling */
  height: 20px; /* Set the height of the progress bar */
  border-radius: 5px; /* Round the corners of the progress bar */
  background-color: #f3f3f3; /* Light grey background */
  width: 100%;
}
/* Style the progress bar container */
.progress-container {
  margin-left: 20px;
  margin-right: 20px;
  flex-grow: 1; /* Allow the progress container to take up remaining space */
}
/* Set the color of the progress bar fill */
progress::-webkit-progress-value {
  background-color: #3498db; /* Blue color for the fill */
}
progress::-moz-progress-bar {
  background-color: #3498db; /* Blue color for the fill in Firefox */
}
/* Style the text on the progress bar */
progress::after {
  content: attr(value '%'); /* Display the progress value followed by '%' */
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white; /* Set text color */
  font-size: 14px; /* Set font size */
}
/* Style other texts */
.loader-container > span {
  margin-left: 5px; /* Add spacing between the progress bar and the text */
}
.progress-bar > .generating {
  display: none !important;
}
.progress-bar{
  height: 30px !important;
}
.hint-container > .generating {
  display: none !important;
}
.hint-container{
  height: 150px !important;
}
.json-container{
  height: 600px;
  overflow: auto !important;
}
.type_row{
  height: 150px !important;
}
.type_small_row{
  height: 100px !important;
}
.scroll-hide{
  resize: auto !important;
}
.refresh_button{
  border: none !important;
  background: none !important;
  font-size: none !important;
  box-shadow: none !important;
}
.element1 {
  opacity: 0.01;
}
#inpaint_sketch { overflow: overlay !important; resize: auto; background: var(--panel-background-fill); z-index: 5; }
/* Custom CSS for mobile-friendliness */
@media (max-width: 768px) {
  body, .gr-textbox, .gr-button {
    font-size: 14px;
  }
  #component-321
  {
    padding: unset;
  }
  #main_view {
    height: 100% !important;
  }
  #component-19{
    padding: 0px 6% 0 7%;
  }
  #component-22{
    margin-top: 55px;
  }
}
"""
progress_html = """
<div class="loader-container">
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
"""
scripts = """
function generate_shortcut(){
  document.addEventListener('keydown', (e) => {
    let handle = 'none';
    if (e.key !== undefined) {
      if ((e.key === 'Enter' && e.ctrlKey)) handle = 'run';
    } else if (e.keyCode !== undefined) {
      if ((e.keyCode === 13 && e.ctrlKey)) handle = 'run';
    }
    if (handle == 'run') {
      const button = document.getElementById('generate');
      if (button) button.click();
      e.preventDefault();
    }
  });
}
generate_shortcut();
"""

from shared import state


def make_progress_html(number, text):
    if number == -1:
        number = state["last_progress"]
    else:
        state["last_progress"] = number
    return progress_html.replace("*number*", str(number)).replace("*text*", text)
