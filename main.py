import cv2
import mediapipe as mp
import pyautogui
import time

from fontTools.merge.util import current_time

# MediaPipe modüllerini başlat
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# PARMAK YAKINLIKLARINI HESABA KATIP BİR FONKSİYON DA YAZABİLİRİM
# ÖRNEK OLARAK BAŞ VE SERÇE OLDUĞUDNA BİR TUŞ ÇALIŞSIN 4 20
# VEYA SADECE ORTA İLE YÜZÜK YAKIN OLDUĞUNDA ÇALIŞSIN 12 16
def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    closed_fingers = 0
    fingertip_ids = [4, 8, 12, 16, 20]

    for tip_id in fingertip_ids:
        tip = hand_landmarks.landmark[tip_id]
        if abs(tip.x - wrist.x) < 0.1 and abs(tip.y - wrist.y) < 0.1:
            closed_fingers += 1

    return closed_fingers >= 4
# Başlangıçta son komut zamanı
last_trigger_time = 0
cooldown_duration = 3  # saniye
# Kamera başlat
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Kameradan görüntü alınamadı.")
                break

            # BGR'den RGB'ye çevir
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Performans artırmak için yazma koruması kapat
            image.flags.writeable = False

            # Tespiti yap
            results = holistic.process(image)

            # Görüntüyü tekrar yazılabilir hale getir
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- YÜZ ---
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

            # --- VÜCUT DURUŞU (POSE) ---
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style()
                )

            # --- SAĞ EL ---
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style()
                )

            # --- SOL EL ---
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style()
                )

            # Görüntüyü göster
            cv2.imshow("MediaPipe Holistic - Face, Pose, Hands", image)
            if results.right_hand_landmarks:
                current_time=time.time();
                if is_fist(results.right_hand_landmarks)& (current_time - last_trigger_time >= cooldown_duration):
                    print("Yumruk algılandı! Alt+F4 gönderiliyor...")
                    pyautogui.hotkey('alt', 'f4')
                    last_trigger_time=current_time

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break





# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
