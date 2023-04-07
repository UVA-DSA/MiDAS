function sendData() {
    var subject = document.getElementById("subject").value;
    var trial = document.getElementById("trial").value;
    var task = document.getElementById("task").value;
    var rate = document.getElementById("rate").value;

    if (!subject || !trial || !task || !rate) {
        alert("Please fill out all forms");
        return false;
      } else {
        eel.sendData(subject, trial, task, rate)();
      }
}