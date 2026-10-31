# Human Activity Recognition with a Hidden Markov Model (HMM)

Formative 2 – Machine Learning Techniques II

We record phone sensor data (accelerometer and gyroscope) with the Sensor Logger app and build an
HMM from scratch with NumPy to guess the activity: **standing, walking, jumping, still**.

## Videos

Standing: [https://youtube.com/shorts/mh4GCAZ88eY] · Walking: [https://youtube.com/shorts/jWPxOcB7qUQ] · Jumping: [https://youtube.com/shorts/qXSUvkeXHU8] · Still: [https://youtube.com/shorts/_UdYE-N8nIA]

## Folders

```
data/        original Sensor Logger zip files (one folder for each activity)
data_csv/    clean 50 Hz CSV files + sampling_rates.csv (made by the notebook)
notebook/    hmm_activity_recognition.ipynb
results/     figures and tables
```

## Data

200 recordings (50 for each activity), each about 5–6 seconds. CSV columns:
`time_s`, `ax`, `ay`, `az` (accelerometer), `gx`, `gy`, `gz` (gyroscope).

## Method

1. Make CSV files and change all data to 50 Hz
2. Cut into windows of 2 seconds (new window every 1 second)
3. Make 6 features per window (time: `acc_mean`, `acc_std`, `acc_sma`, `gyr_std`; FFT: `dom_freq`, `energy`)
4. Z-score the features, keep the last 2 recordings of each activity as unseen test data
5. HMM with Baum-Welch (training) and Viterbi (prediction)

## Results

Test accuracy **100%** (30 of 30 windows), training accuracy 95.8%. Sensitivity and specificity are 1.00
for all activities. The test data comes from the same phone, person and day as the training data.

## Run

```bash
pip install numpy pandas matplotlib seaborn jupyter
cd notebook
jupyter notebook hmm_activity_recognition.ipynb   # then click Run all
```