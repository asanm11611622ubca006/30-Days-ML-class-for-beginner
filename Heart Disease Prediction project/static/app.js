/**
 * Heart Disease Prediction - Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultCard = document.getElementById('resultCard');
    const resultContent = document.getElementById('resultContent');

    // Form submission
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Show loading state
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        submitBtn.disabled = true;

        // Collect form data
        const formData = {
            age: document.getElementById('age').value,
            sex: document.getElementById('sex').value,
            chestPainType: document.getElementById('chestPainType').value,
            restingBP: document.getElementById('restingBP').value,
            cholesterol: document.getElementById('cholesterol').value,
            fastingBS: document.getElementById('fastingBS').value,
            restingECG: document.getElementById('restingECG').value,
            maxHR: document.getElementById('maxHR').value,
            exerciseAngina: document.getElementById('exerciseAngina').value,
            oldpeak: document.getElementById('oldpeak').value,
            stSlope: document.getElementById('stSlope').value
        };

        try {
            // Send data to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (response.ok) {
                displayResult(result);
            } else {
                displayError(result.error || 'An error occurred during prediction');
            }
        } catch (error) {
            displayError('Failed to connect to the server. Please try again.');
            console.error('Error:', error);
        } finally {
            // Reset button state
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    function displayResult(result) {
        const isDisease = result.prediction === 1;
        const diseaseProb = result.probability.disease;
        const noDiseaseProb = result.probability.no_disease;

        const resultHTML = `
            <div class="result-icon">
                ${isDisease ? '⚠️' : '✅'}
            </div>
            <div class="result-message ${isDisease ? 'result-negative' : 'result-positive'}">
                ${result.message}
            </div>
            <div class="result-details">
                <h3 style="margin-bottom: 20px; color: #333;">Prediction Confidence:</h3>
                
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>No Heart Disease</span>
                        <span>${noDiseaseProb.toFixed(2)}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill positive" style="width: ${noDiseaseProb}%">
                            ${noDiseaseProb.toFixed(1)}%
                        </div>
                    </div>
                </div>

                <div class="probability-bar">
                    <div class="probability-label">
                        <span>Heart Disease</span>
                        <span>${diseaseProb.toFixed(2)}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill negative" style="width: ${diseaseProb}%">
                            ${diseaseProb.toFixed(1)}%
                        </div>
                    </div>
                </div>

                <p style="margin-top: 20px; color: #666; font-size: 14px; line-height: 1.6;">
                    ${isDisease
                ? '⚠️ <strong>Warning:</strong> The model predicts a high likelihood of heart disease. Please consult with a healthcare professional for proper diagnosis and treatment.'
                : '✅ <strong>Good News:</strong> The model predicts a low likelihood of heart disease. However, maintain a healthy lifestyle and regular check-ups.'}
                </p>
            </div>
        `;

        resultContent.innerHTML = resultHTML;
        resultCard.style.display = 'block';

        // Smooth scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function displayError(message) {
        const errorHTML = `
            <div class="result-icon">❌</div>
            <div class="result-message" style="color: #e74c3c;">
                Prediction Failed
            </div>
            <div class="result-details">
                <p style="color: #666;">${message}</p>
            </div>
        `;

        resultContent.innerHTML = errorHTML;
        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Input validation helpers
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function () {
            if (this.value < parseFloat(this.min)) {
                this.value = this.min;
            }
            if (this.value > parseFloat(this.max)) {
                this.value = this.max;
            }
        });
    });
});
