import cv2
import winsound
from deepface import DeepFace

EMOTION_TONES = {
    "happy": 1000,
    "sad": 400,
    "angry": 700,
    "neutral": 600,
    "fear": 300,
    "surprise": 1200,
    "disgust": 500
}

EMOTION_COLORS = {
    "happy":  (0, 255, 0),
    "sad":    (255, 0, 0),
    "angry":  (0, 0, 255),
    "neutral": (200, 200, 200),
    "fear":   (100, 0, 100),
    "surprise": (0, 255, 255),
    "disgust": (0, 150, 0)
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

last_emotion = None  # to avoid sound spam

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Frame not captured!")
        continue

    # ---------------------------------------
    # DeepFace Analysis
    # ---------------------------------------
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # DeepFace sometimes returns list
    if isinstance(results, list):
        results = results[0]

    emotion = results["dominant_emotion"]
    confidence_dict = results["emotion"]  # full emotion scores

    # ---------------------------------------
    # PLAY TONE ONLY WHEN EMOTION CHANGES
    # ---------------------------------------
    if emotion != last_emotion:
        freq = EMOTION_TONES.get(emotion, 800)
        winsound.Beep(freq, 250)  # smooth tone
        last_emotion = emotion

    # ---------------------------------------
    # UI: Emotion Text + Confidences
    # ---------------------------------------
    overlay = img.copy()

    # Background color bar
    color = EMOTION_COLORS.get(emotion, (0, 0, 0))
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 100), color, -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    cv2.putText(img, f'Emotion: {emotion.upper()}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # ---------------------------------------
    # UI: Emotion Confidence Bars
    # ---------------------------------------
    bar_x = 20
    bar_y = 120
    
    for emo, val in confidence_dict.items():
        width = int(val) * 2  # scale
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + width, bar_y + 25), (0, 255, 255), -1)
        cv2.putText(img, f"{emo}: {val:.1f}%", (bar_x + 220, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        bar_y += 35

    # ---------------------------------------
    # Display Frame
    # ---------------------------------------
    cv2.imshow(" Advanced Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
