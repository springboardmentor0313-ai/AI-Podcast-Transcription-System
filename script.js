function uploadAudio() {
  const fileInput = document.getElementById("audioFile");
  const status = document.getElementById("status");

  if (!fileInput.files.length) {
    alert("Please select an audio file");
    return;
  }

  const formData = new FormData();
  formData.append("audio", fileInput.files[0]);

  status.innerText = "Processing... Please wait";

  fetch("/upload", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    status.innerText = "";
    document.getElementById("main").style.display = "block";

    const audio = document.getElementById("audio");
    audio.src = URL.createObjectURL(fileInput.files[0]);

    const segmentsDiv = document.getElementById("segments");
    segmentsDiv.innerHTML = "";

    data.forEach(seg => {
      const div = document.createElement("div");
      div.className = "segment";
      div.innerHTML = `
        <h3>${seg.title}</h3>
        <p>${seg.start}s - ${seg.end}s</p>
        <p>${seg.summary.substring(0, 80)}...</p>
        <button onclick="playSegment(${seg.start}, '${seg.text.replace(/'/g, "")}')">▶ Play Segment</button>
      `;
      segmentsDiv.appendChild(div);
    });
  });
}

function playSegment(time, text) {
  const audio = document.getElementById("audio");
  audio.currentTime = time;
  audio.play();
  document.getElementById("segmentText").innerText = text;
}
