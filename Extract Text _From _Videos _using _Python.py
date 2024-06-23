import speech_recognition as sr 
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pytesseract
import cv2

def extract_text_from_frames(video_path, output_text_file):
    num_seconds_video = 52 * 60
    print("The video is {} seconds".format(num_seconds_video))
    chunk_seconds = 60
    l = list(range(0, num_seconds_video + 1, chunk_seconds))

    diz = {}
    for i in range(len(l) - 1):
        ffmpeg_extract_subclip(video_path, max(0, l[i] - 2 * (l[i] != 0)), l[i + 1],
                               targetname="chunks/cut{}.mp4".format(i + 1))
        clip = mp.VideoFileClip(r"chunks/cut{}.mp4".format(i + 1))
        clip.audio.write_audiofile(r"converted/converted{}.wav".format(i + 1))

        r = sr.Recognizer()
        audio = sr.AudioFile("converted/converted{}.wav".format(i + 1))
        with audio as source:
            r.adjust_for_ambient_noise(source)
            audio_file = r.record(source)
        result = r.recognize_google(audio_file)
        diz['chunk{}'.format(i + 1)] = result

        cap = cv2.VideoCapture(r"chunks/cut{}.mp4".format(i + 1))
        frames_text = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            frames_text.append(text)
        
        diz['frames_chunk{}'.format(i + 1)] = '\n'.join(frames_text)
        cap.release()

    text_chunks = [diz['chunk{}'.format(i + 1)] for i in range(len(diz))]
    frames_text_chunks = [diz['frames_chunk{}'.format(i + 1)] for i in range(len(diz))]
    
    with open(output_text_file, mode='w') as file:
        file.write("Recognized Speech:\n")
        file.write("\n".join(text_chunks))
        file.write("\n\nOCR from Video Frames:\n")
        file.write("\n".join(frames_text_chunks))

    print("Text extraction complete. Check '{}' for results.".format(output_text_file))

video_path = "videorl.mp4"
output_text_file = "recognized.txt"
extract_text_from_frames(video_path, output_text_file)
