## Overview
Detect when two people (in a group) are looking at each other while talking

* Area of interest:
    * face and body
    * gazes
* Input:
    * 4-5 videos and probaly 1 audio file
    * a tsv file containing the gaze timestamps of each person in the group, estimating gaze direction
    * configuration values
* Output:
    * [ELAN](https://archive.mpi.nl/tla/elan)-compatible file
        * Tier / start time / end time / duration / label (e.g. gaze with user i)


## GUI requirements
### Video (& audio) importing
Allow users to import multiple video files and optionally an audio file.
Possible steps:

1. User clicks "Import" button
2. Define number of tracked people (e.g. 4)
    * Assign each person a unique ID (e.g. 1, 2, 3, 4)
3. For each person, select the corresponding video file (what they saw) and the tsv file (gaze direction estimation)
4. Optionally select the video file that captures the group interaction (e.g. a wide shot of all participants)
5. Optionally select an audio file that records the group conversation
6. Validate (in the background):
    * the number of video files matches the number of tracked people
    * the video files are compatible (e.g. same frame rate, resolution, etc.)
    * If validation fails, show an error message and allow the user to re-import files.
7. User clicks "OK" button to confirm the import
    * User clicks "Cancel" button to abort the import process

### Video preprocessing
For synchronization between videos

* This step is mandatory
* Automatically detect the onset and offset frames of each video, determined when the board was clapped
* Allow users to manually adjust the onset and offset frames if necessary by marking the frames in the GUI

For synchronization between video and audio, interms of gaze timestamps

* User can enable/disable this optional step
* When enabled, the application should:
    * Fetch the onset and offset timestamps of the video files from the previous preprocessing step
    * Automatically detect the onset and offset timestamps of the audio file, determined when the board was clapped
    * Calculate the mean values
    * Overwrite the calculated timestamps in the gaze tsv files with the mean values, so that the gaze timestamps are synchronized with the video and audio files
* When disabled, we assume that the video and audio files are already synchronized, and we use the onset and offset timestamps of the video files from the previous preprocessing step.

### Video processing
#### Before processing
* User can define settings for detection pipeline:
    * (Mandatory) Object detection
        * model
        * reID
        * tracker
        * object classes
        * embeddings per track
    * (Optional) Face detection
        * model
        * detection frame size
        * detection threshold
        * embeddings per track
    * (Optional) Body pose detection
        * model
        * confidence threshold
* User can define settings for the output file:
    * Output file name & format
    * Fields to include in the output file, e.g.
        * tier
        * start time
        * end time
        * duration
        * label (e.g. gaze with user i)
#### During processing
* Display a progress bar indicating the processing status
* Allow users to pause or cancel the processing if needed
#### After processing
Display the results of the processing on videos, including:
* Detected objects, faces, and body poses
* Gaze directions and interactions between tracked people via highlight video(s), where:
    * The person in the video is talking
    * Mutual gaze is detected between two people in the video

### Output exporting
Allow users to export the results in:

* an [ELAN](https://archive.mpi.nl/tla/elan)-compatible file format, and/or other formats (e.g. CSV)
* videos with annotations overlay, showing detected objects, faces, and body poses


## Software pipeline
Based on the general goal and the GUI requirements, the software pipeline might consists of the following steps:
1. Video (& audio) importing (as described in the GUI requirements)
2. Video preprocessing (as described in the GUI requirements)
3. **Video processing**
    * **Object detection** (e.g. with BoxMOT): detect objects in each video frame, and assign unique IDs to each detected object
        * ***Clustering to yield tracklets for each tracked person (?, not sure)***
    * **Face landmark extraction**: detect faces inside each tracked object box (aka. person) and yield:
        * frame and video frame index
        * `track_id` and `person_id`
        * tracklet box coordinates
        * detected face box and confidence score
        * facial landmarks: `left_eye`, `right_eye`, `nose`, `left_mouth_corner`, `right_mouth_corner`
    * Make sure the gaze timestamps in the tsv files are synchronized with the video and audio files (see video preprocessing step)
    * **Gaze-on-face assignment**: for each gaze timestamp in the tsv files, assign the gaze to a detected face in the corresponding video frame, and append the tsv file with the following columns::
        * `gaze_on_face` (boolean): 1 if the gaze is on a detected face, 0 otherwise
        * `gaze_on_body` (boolean): 1 if the gaze is on a detected body, 0 otherwise
        * `gaze_person_id`: the ID of the person being gazed at, if any
        * `gaze_track_id`: the ID of the tracklet being gazed at
        * `face_x1`, `face_y1`, `face_x2`, `face_y2`: the coordinates of the detected face box, if any
        * `nearest_face_feature`: the nearest facial feature to the gaze point, if any
        * `nearest_face_feature_distance`: the distance from the gaze point to the nearest facial feature, if any
        * `nearest_body_feature`: the nearest body feature to the gaze point, if any, e.g. `left`, `right`, `top`, `bottom`, `center`
        * `nearest_body_feature_distance`: the distance from the gaze point to the nearest body feature, if any
    * **Mutual gaze detection**: from all updated tsv files, detect mutual gaze between tracked people, and create a new ELAN-compatible file with the following columns:
        * Need to discuss: (A) one mutual gaze file for all tracked people (group video will be used?), or (B) mutual gaze as additional info for each tracked people, i.e. each video?
        * `tier`: e.g. `user_j`
        * `start_time`: the start time of the mutual gaze event
        * `end_time`: the end time of the mutual gaze event
        * `duration`: the duration of the mutual gaze event
        * `label`: e.g. `mg_w_user_i`

4. Output exporting (as described in the GUI requirements)
