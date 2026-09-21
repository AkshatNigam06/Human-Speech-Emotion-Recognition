const audioFile = document.getElementById("audioFile");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");

const emotion = document.getElementById("emotion");
const confidence = document.getElementById("confidence");
const probabilities = document.getElementById("probabilities");


// Show selected file name
audioFile.addEventListener("change", function () {

    if (audioFile.files.length > 0) {
        fileName.textContent = audioFile.files[0].name;
    } else {
        fileName.textContent = "No file selected";
    }

});


// Analyze audio
analyzeBtn.addEventListener("click", async function () {

    if (audioFile.files.length === 0) {
        alert("Please select a WAV audio file first.");
        return;
    }

    const formData = new FormData();

    formData.append("audio", audioFile.files[0]);

    emotion.textContent = "Analyzing...";
    confidence.textContent = "—";
    probabilities.innerHTML = "";

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        // Show emotion
        emotion.textContent = data.emotion;

        // Show confidence
        confidence.textContent = data.confidence + "%";


        // Show emotion probabilities
        for (const [name, value] of Object.entries(data.probabilities)) {

            const row = document.createElement("div");

            row.innerHTML = `
                <p>${name}: ${value}%</p>

                <div class="probability-bar">
                    <div 
                        class="probability-fill" 
                        style="width: ${value}%">
                    </div>
                </div>
            `;

            probabilities.appendChild(row);
        }


        console.log("Prediction:", data);

    } catch (error) {

        console.error(error);

        emotion.textContent = "Error";
        confidence.textContent = "—";
        probabilities.innerHTML = "";

        alert("Could not connect to Flask server.");

    }

});