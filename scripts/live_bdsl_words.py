
import cv2
from ultralytics import YOLO
from collections import deque, Counter

MODEL_PATH = "runs/classify/train-3/weights/best.pt"

print("Loading BdSL Word Recognition Model...")
model = YOLO(MODEL_PATH)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

history = deque(maxlen=10)
frame_count = 0
top5 = []

print("Webcam started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Run prediction every 3 frames
    if frame_count % 3 == 0:
        results = model.predict(
            source=frame,
            imgsz=160,
            device="cpu",
            verbose=False
        )

        probabilities = results[0].probs
        best_id = int(probabilities.top1)
        confidence = float(probabilities.top1conf)

        predicted_word = model.names[best_id]

        if confidence >= 0.25:
            history.append(predicted_word)
        else:
            history.append("Uncertain")

        top5 = []

        for class_id, score in zip(
            probabilities.top5,
            probabilities.top5conf
        ):
            top5.append(
                (model.names[int(class_id)], float(score))
            )

    # Stable prediction
    stable_word = "Waiting..."

    valid_predictions = [
        word for word in history if word != "Uncertain"
    ]

    if valid_predictions:
        counts = Counter(valid_predictions)
        stable_word, votes = counts.most_common(1)[0]

        if votes < 6:
            stable_word = "Unstable"

    cv2.rectangle(frame, (10, 10), (520, 260), (25, 25, 25), -1)

    cv2.putText(
        frame, "BdSL WORD RECOGNITION",
        (25, 42), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    cv2.putText(
        frame, f"Prediction: {stable_word}",
        (25, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2
    )

    if top5:
        cv2.putText(
            frame, "TOP 5 PREDICTIONS:",
            (25, 135), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        for i, (word, score) in enumerate(top5):
            text = f"{i+1}. {word}: {score*100:.1f}%"

            cv2.putText(
                frame, text,
                (25, 165 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 220, 255), 1
            )

    cv2.imshow("Global Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
