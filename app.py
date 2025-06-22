# from flask import Flask, request, send_file, jsonify, render_template, redirect, url_for
# import os
# from model_runner import run_model  # Make sure this returns full path of .mp4 output

# app = Flask(__name__)

# UPLOAD_FOLDER = './uploads'
# OUTPUT_FOLDER = './result'  # Changed to 'result'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')  # Renders HTML form

# # Updated route for '/inference'
# @app.route('/inference', methods=['GET', 'POST'])
# def inference():
#     if request.method == 'POST':
#         # 1. Validate video upload
#         if 'video' not in request.files or request.files['video'].filename == '':
#             return jsonify({'error': '❌ No video file uploaded.'}), 400

#         # 2. Validate text query
#         text = request.form.get('text', '').strip()
#         if not text:
#             return jsonify({'error': '❌ No text query provided.'}), 400

#         # 3. Save input files
#         video = request.files['video']
#         video_filename = video.filename  # Store the original video filename
#         video_path = os.path.join(UPLOAD_FOLDER, video_filename)
#         video.save(video_path)

#         query_path = os.path.join(UPLOAD_FOLDER, 'query.txt')
#         with open(query_path, 'w') as f:
#             f.write(text)

#         # 4. Run model inference
#         try:
#             # Pass the dynamic video path and output directory
#             output_path = run_model(video_path, query_path, output_dir=OUTPUT_FOLDER)
#         except Exception as e:
#             print(f"[ERROR] Model inference failed: {str(e)}")
#             return jsonify({'error': '❌ Model inference failed. Check logs.'}), 500

#         # 5. Check and send output video
#         if not os.path.exists(output_path) or not output_path.endswith('.mp4'):
#             return jsonify({'error': '❌ Output video not generated.'}), 500

#         # Send the result video to the client by rendering result.html
#         # Make sure we pass the correct output path to the template
#         return render_template('result.html', video_path=output_path)

#     return render_template('inference.html')  # Render the inference form on GET request


# if __name__ == "__main__":
#     app.run(debug=True)















####the below one needs to be uncommented####



# from flask import Flask, request, send_file, jsonify, render_template, redirect, url_for
# import os
# from threading import Thread
# from model_runner import run_model  # Make sure this returns full path to .mp4 output

# app = Flask(__name__)

# UPLOAD_FOLDER = './uploads'
# OUTPUT_FOLDER = './result'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Track inference status
# inference_status = {
#     "status": "idle",
#     "video_path": ""
# }

# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')  # Form to upload video & text









####the above one needs to be uncommented####








# @app.route('/inference', methods=['GET', 'POST'])
# def inference():
#     global inference_status

#     if request.method == 'POST':
#         if 'video' not in request.files or request.files['video'].filename == '':
#             return jsonify({'error': '❌ No video file uploaded.'}), 400

#         text = request.form.get('text', '').strip()
#         if not text:
#             return jsonify({'error': '❌ No text query provided.'}), 400

#         # Save uploaded files
#         video = request.files['video']
#         video_filename = video.filename
#         video_path = os.path.join(UPLOAD_FOLDER, video_filename)
#         video.save(video_path)

#         query_path = os.path.join(UPLOAD_FOLDER, 'query.txt')
#         with open(query_path, 'w') as f:
#             f.write(text)















####the below one needs to be uncommented####








# @app.route('/inference', methods=['GET', 'POST'])
# def inference():
#     global inference_status

#     if request.method == 'POST':
#         if 'video' not in request.files or request.files['video'].filename == '':
#             return jsonify({'error': '❌ No video file uploaded.'}), 400

#         text = request.form.get('text', '').strip()
#         if not text:
#             return jsonify({'error': '❌ No text query provided.'}), 400

#         # Save uploaded files
#         video = request.files['video']
#         video_filename = video.filename
#         video_path = os.path.join(UPLOAD_FOLDER, video_filename)
#         video.save(video_path)

#         query_path = os.path.join(UPLOAD_FOLDER, 'query.txt')
#         with open(query_path, 'w') as f:
#             f.write(text)

#         # Background inference thread
#         def background_task():
#             global inference_status
#             inference_status["status"] = "processing"
#             try:
#                 # Run the model using the correct command (update the arguments as needed)
#                 output_path = run_model(
#                     video_path,
#                     query_path,
#                     output_dir=OUTPUT_FOLDER,
#                     backbone="video-swin-b",  # Backbone to use
#                     bpp="/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth",  # Path to backbone
#                     ckpt="/home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar"  # Checkpoint path
#                 )
#                 inference_status["status"] = "done"
#                 inference_status["video_path"] = output_path
#             except Exception as e:
#                 inference_status["status"] = "error"
#                 print(f"[ERROR] Inference failed: {e}")

#         Thread(target=background_task).start()
#         return redirect(url_for('progress'))

#     return render_template('inference.html')








####the above one needs to be uncommented####


    #     # Background inference thread
    #     def background_task():
    #         global inference_status
    #         inference_status["status"] = "processing"
    #         try:
    #             output_path = run_model(video_path, query_path, output_dir=OUTPUT_FOLDER)
    #             inference_status["status"] = "done"
    #             inference_status["video_path"] = output_path
    #         except Exception as e:
    #             inference_status["status"] = "error"
    #             print(f"[ERROR] Inference failed: {e}")

    #     Thread(target=background_task).start()
    #     return redirect(url_for('progress'))

    # return render_template('inference.html')





















####the below one needs to be uncommented####






# @app.route('/progress')
# def progress():
#     return render_template('progress.html')

# @app.route('/check_status')
# def check_status():
#     global inference_status
#     return jsonify(inference_status)

# @app.route('/view_result')
# def view_result():
#     video_path = request.args.get('video_path')
#     return render_template('result.html', video_path=video_path)

# # Optional direct download route
# @app.route('/download/<path:filename>')
# def download_file(filename):
#     return send_file(filename, as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True,port = 5002)












####the above one needs to be uncommented####
































































from flask import Response, request, send_file
import mimetypes
import re

from flask import Flask, request, send_file, jsonify, render_template, redirect, url_for, send_from_directory
import os
import shutil
from threading import Thread
from werkzeug.utils import secure_filename
from model_runner import run_model  # This should return full path to the .mp4 output
from flask_cors import CORS


app = Flask(__name__, static_folder='static')

CORS(app)
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './result'
STATIC_OUTPUT_FOLDER = './static/result'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_OUTPUT_FOLDER, exist_ok=True)

# Track inference status
inference_status = {
    "status": "idle",
    "video_path": ""
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Form to upload video & text

@app.route('/result/<path:filename>')
def stream_video(filename):
    file_path = os.path.join("static", "result", filename)  # Correct path for static directory
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(file_path, mimetype='video/mp4')

    # Range handling
    size = os.path.getsize(file_path)
    byte1, byte2 = 0, None

    # Match the range header
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if m:
        g = m.groups()
        byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1

    with open(file_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    # Return partial content response with correct headers
    rv = Response(data,
                  206,
                  mimetype='video/mp4',
                  content_type='video/mp4',
                  direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))  # Content-Length is important for the video stream
    return rv


@app.route('/download/<path:filename>')
def download_segmented_video(filename):  # Changed name to avoid conflict
    file_path = os.path.join("static", "result", filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    global inference_status

    if request.method == 'POST':
        if 'video' not in request.files or request.files['video'].filename == '':
            return jsonify({'error': '❌ No video file uploaded.'}), 400

        text = request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': '❌ No text query provided.'}), 400

        # Save uploaded video
        video = request.files['video']
        video_filename = video.filename
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video.save(video_path)

        # Save text query
        query_path = os.path.join(UPLOAD_FOLDER, 'query.txt')
        with open(query_path, 'w') as f:
            f.write(text)

        # Background inference thread
        def background_task():
            global inference_status
            inference_status["status"] = "processing"
            try:
                # Run the model
                output_path = run_model(
                    video_path,
                    query_path,
                    output_dir=OUTPUT_FOLDER,
                    backbone="video-swin-b",
                    bpp="/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth",
                    ckpt="/home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar"
                )

                # Move output to static/result
                # output_filename = os.path.basename(output_path)
                # static_output_path = os.path.join(STATIC_OUTPUT_FOLDER, output_filename)
                # shutil.copy(output_path, static_output_path)

                # # Update web-accessible path
                # inference_status["video_path"] = f"/static/result/{output_filename}"
                
                
                
                output_filename = os.path.basename(output_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g., "CSE_ADMIN_VID"

                # Create nested static directory: ./static/result/<video_name>/SOC/
                nested_static_dir = os.path.join(STATIC_OUTPUT_FOLDER, video_name, "SOC")
                os.makedirs(nested_static_dir, exist_ok=True)

                # Define new static output path
                static_output_path = os.path.join(nested_static_dir, output_filename)

                # Copy output file to static path
                shutil.copy(output_path, static_output_path)

                # Update web-accessible path
                inference_status["video_path"] = f"/result/{video_name}/SOC/{output_filename}"
                inference_status["status"] = "done"


            except Exception as e:
                inference_status["status"] = "error"
                print(f"[ERROR] Inference failed: {e}")

        Thread(target=background_task).start()
        return redirect(url_for('progress'))

    return render_template('inference.html')


@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/progress')
def progress():
    return render_template('progress.html')

@app.route('/check_status')
def check_status():
    global inference_status
    return jsonify(inference_status)

@app.route('/view_result')
def view_result():
    video_path = request.args.get('video_path')
    return render_template('result.html', video_path=video_path)

# Optional direct download route
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
