
const audioFile = document.getElementById("audioFile");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");

const startRecordBtn = document.getElementById("startRecordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const analyzeRecordBtn = document.getElementById("analyzeRecordBtn");

const recordingStatus = document.getElementById("recordingStatus");
const recordingTime = document.getElementById("recordingTime");
const audioPreview = document.getElementById("audioPreview");

const emotionElement = document.getElementById("emotion");
const confidenceElement = document.getElementById("confidence");

let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let recordedAudioURL = null;
let mediaStream = null;
let timerInterval = null;
let secondsElapsed = 0;


// ==========================================
// FILE UPLOAD
// ==========================================

audioFile.addEventListener("change", () => {
    const file = audioFile.files[0];

    fileName.textContent = file
        ? file.name
        : "No file selected";
});


// ==========================================
// CLEAR AUDIO PREVIEW
// ==========================================

function clearAudioPreview() {
    audioPreview.pause();
    audioPreview.removeAttribute("src");
    audioPreview.load();
    audioPreview.hidden = true;

    if (recordedAudioURL) {
        URL.revokeObjectURL(recordedAudioURL);
        recordedAudioURL = null;
    }
}


// ==========================================
// RECORDING TIMER
// ==========================================

function startTimer() {
    secondsElapsed = 0;
    recordingTime.textContent = "00:00";

    timerInterval = setInterval(() => {
        secondsElapsed++;

        const minutes = String(
            Math.floor(secondsElapsed / 60)
        ).padStart(2, "0");

        const seconds = String(
            secondsElapsed % 60
        ).padStart(2, "0");

        recordingTime.textContent = `${minutes}:${seconds}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
}


// ==========================================
// START RECORDING
// ==========================================

startRecordBtn.addEventListener("click", async () => {
    try {
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {
            throw new Error("Microphone is not supported.");
        }

        clearAudioPreview();

        recordedBlob = null;
        audioChunks = [];

        analyzeRecordBtn.disabled = true;
        startRecordBtn.disabled = true;
        stopRecordBtn.disabled = false;

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        const options = {};

        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
            options.mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/webm")) {
            options.mimeType = "audio/webm";
        }

        mediaRecorder = new MediaRecorder(mediaStream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onerror = (event) => {
            console.error("Recording error:", event.error);
            recordingStatus.textContent = "Recording error!";
        };

        // ==========================================
        // STOP EVENT + WAV PREVIEW CONVERSION
        // ==========================================

        mediaRecorder.onstop = async () => {
            stopTimer();

            const mimeType =
                mediaRecorder.mimeType || "audio/webm";

            recordedBlob = new Blob(audioChunks, {
                type: mimeType
            });

            console.log("Recorded audio size:", recordedBlob.size);
            console.log("Recorded audio type:", recordedBlob.type);
            console.log("Audio chunks:", audioChunks.length);

            if (recordedBlob.size === 0) {
                recordingStatus.textContent =
                    "Recording is empty. Please record again.";

                analyzeRecordBtn.disabled = true;
            } else {
                recordingStatus.textContent =
                    "Converting audio for playback...";

                analyzeRecordBtn.disabled = false;

                try {
                    const previewFormData = new FormData();

                    previewFormData.append(
                        "audio",
                        new File(
                            [recordedBlob],
                            "recording.webm",
                            { type: recordedBlob.type }
                        )
                    );

                    const response = await fetch("/preview", {
                        method: "POST",
                        body: previewFormData
                    });

                    if (!response.ok) {
                        let errorMessage = "Audio conversion failed";

                        try {
                            const error = await response.json();
                            errorMessage = error.error || errorMessage;
                        } catch (_) {}

                        throw new Error(errorMessage);
                    }

                    const wavBlob = await response.blob();

                    if (wavBlob.size === 0) {
                        throw new Error("Converted WAV file is empty.");
                    }

                    clearAudioPreview();

                    recordedAudioURL =
                        URL.createObjectURL(wavBlob);

                    audioPreview.src = recordedAudioURL;
                    audioPreview.hidden = false;
                    audioPreview.controls = true;
                    audioPreview.volume = 1;
                    audioPreview.muted = false;
                    audioPreview.load();

                    audioPreview.onerror = () => {
                        console.error(
                            "WAV playback error:",
                            audioPreview.error
                        );
                    };

                    recordingStatus.textContent =
                        "Recording completed! Play your audio.";

                    console.log("WAV preview ready:", wavBlob.size);

                } catch (error) {
                    console.error("Preview conversion error:", error);

                    recordingStatus.textContent =
                        "Preview failed: " + error.message;
                }
            }

            startRecordBtn.disabled = false;
            stopRecordBtn.disabled = true;

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
        };

        mediaRecorder.start(1000);

        recordingStatus.textContent = "Recording...";
        startTimer();

    } catch (error) {
        console.error("Microphone error:", error);

        recordingStatus.textContent =
            "Microphone error: " + error.message;

        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;
        analyzeRecordBtn.disabled = true;

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        stopTimer();
    }
});


// ==========================================
// STOP RECORDING
// ==========================================

stopRecordBtn.addEventListener("click", () => {
    if (mediaRecorder &&
        mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        recordingStatus.textContent = "Processing recording...";
    }
});


// ==========================================
// ANALYZE AUDIO API
// ==========================================

async function analyzeAudio(file) {
    if (!file) {
        alert("Please select or record an audio file first.");
        return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    try {
        analyzeBtn.disabled = true;
        analyzeRecordBtn.disabled = true;

        recordingStatus.textContent = "Analyzing audio...";

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed");
        }

        displayResults(result);

        recordingStatus.textContent = "Analysis completed!";

    } catch (error) {
        console.error("Analysis error:", error);
        alert("Error: " + error.message);

        recordingStatus.textContent = "Analysis failed.";

    } finally {
        analyzeBtn.disabled = false;
        analyzeRecordBtn.disabled =
            !recordedBlob || recordedBlob.size === 0;
    }
}


// ==========================================
// UPLOAD AUDIO ANALYSIS
// ==========================================

analyzeBtn.addEventListener("click", () => {
    const file = audioFile.files[0];

    if (!file) {
        alert("Please upload a WAV audio file.");
        return;
    }

    analyzeAudio(file);
});


// ==========================================
// RECORDED AUDIO ANALYSIS
// ==========================================

analyzeRecordBtn.addEventListener("click", () => {
    if (!recordedBlob || recordedBlob.size === 0) {
        alert("Please record your voice first.");
        return;
    }

    const recordedFile = new File(
        [recordedBlob],
        "recording.webm",
        { type: recordedBlob.type }
    );

    analyzeAudio(recordedFile);
});


// ==========================================
// DISPLAY PREDICTION RESULTS
// ==========================================

function displayResults(result) {
    emotionElement.textContent = result.emotion;

    confidenceElement.textContent =
        result.confidence + "%";

    const probabilities = result.probabilities || {};

    const probabilityContainer =
        document.getElementById("probabilities");

    if (probabilityContainer) {
        probabilityContainer.innerHTML = "";

        Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1])
            .forEach(([emotion, probability]) => {
                const item = document.createElement("div");
                item.className = "probability-item";

                const label = document.createElement("span");
                label.textContent = emotion;

                const value = document.createElement("span");
                value.textContent = probability + "%";

                const bar = document.createElement("div");
                bar.className = "probability-bar";

                const fill = document.createElement("div");
                fill.className = "probability-fill";
                fill.style.width = probability + "%";

                bar.appendChild(fill);

                item.appendChild(label);
                item.appendChild(value);
                item.appendChild(bar);

                probabilityContainer.appendChild(item);
            });
    }

    document.querySelectorAll(".emotion-card").forEach(card => {
        card.classList.remove("active");

        const cardEmotion = card.dataset.emotion;

        if (cardEmotion === result.emotion) {
            card.classList.add("active");
        }

        const percentage = probabilities[cardEmotion];

        if (percentage !== undefined) {
            const paragraph = card.querySelector("p");

            if (paragraph) {
                paragraph.textContent = percentage + "%";
            }
        }
    });
}


// ==========================================
// CLEANUP
// ==========================================

window.addEventListener("beforeunload", () => {
    stopTimer();

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }

    if (recordedAudioURL) {
        URL.revokeObjectURL(recordedAudioURL);
    }
});
