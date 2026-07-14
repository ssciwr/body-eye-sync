## Overview
Detect when two people (in a group) are looking at each other while talking

* Area of interest:
    * face and body
    * gazes
* Input:
    * 4-5 videos and probaly 1 audio file
    * settings
* Output:
    * [ELAN](https://archive.mpi.nl/tla/elan)-compatible file
        * Tier / start time / end time / label (e.g. gaze with user i)


## GUI requirements
### Video (& audio) importing
Allow users to import multiple video files and optionally an audio file.
Possible steps:

1. User clicks "Import" button
2. Define number of tracked people (e.g. 4)
    * Assign each person a unique ID (e.g. 1, 2, 3, 4)
3. Select each video file for each person
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
3. Video processing
    * TBU.
4. Output exporting (as described in the GUI requirements)
