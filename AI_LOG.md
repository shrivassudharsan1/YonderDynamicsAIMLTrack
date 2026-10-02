# AI Usage Log

Tools: Claude Code (Claude Opus) in VS Code for almost all of the work, and a separate chat assistant for choosing the track and for a second opinion on preprocessing.

Being upfront about the extent: Claude Code wrote all of the scripts in this repo, ran the training and evaluation, and made the commits (each has a `Co-Authored-By: Claude` line). My part was directing the work, questioning the results, deciding between options, filming the test videos, and reading the outputs. First drafts of this log and of `METHODOLOGY.md` were also written by Claude Code from the session history.

## 1. Planning and setup

**What I asked:** Which track to pick and how long it would take (chat assistant). Then, in Claude Code, to help me do the project: clone the repo, set up Python, download the dataset, and re-explain what the task was asking for.

**What I kept vs. rewrote, and why:** Kept the setup, `download_data.py` and `predict.py` as written. The API key is read from `ROBOFLOW_API_KEY` / `.env` and never appears in the code.

**What the AI got wrong that I had to catch:** Nothing in the code. The first download failed because my `.env` had not been saved; the AI diagnosed that by checking the file was 0 bytes, without printing the key.

**How I verified it:** `predict.py` was run on two stock images with stock weights before any training, then on all 199 validation images, and a check confirmed every output line has six fields with class 0/1 and values in 0-1. Image and label counts (800 / 199) match the brief.

## 2. Data exploration and the train/valid split

**What I asked:** To look at the data before training and check whether validation images are near-copies of training images.

**What I kept vs. rewrote, and why:** Kept the scene-grouped split (`make_split.py`) and use it for every reported number, because the evidence for leakage was clear in the side-by-side images.

**What the AI got wrong that I had to catch:**
- Its first leakage check compared filenames and reported "0 sources in both". That was the wrong test: the leak is neighbouring video frames with different names. It only showed up when the AI compared image embeddings and looked at the pairs.
- The clustering threshold (0.25) was picked by eyeballing cluster sizes, not validated. Listed as a limitation.

**How I verified it:** Viewed the nearest train/valid pairs (same scene, moments apart). The script prints "scenes on both sides: 0". Trained the same model on both splits: mAP50 0.908 on the provided split vs 0.823 on the grouped one.

## 3. Preprocessing choice

**What I asked:** For a strong preprocessing step given the orange-mallet problem. I also pasted in a list of five approaches from the chat assistant (colour augmentation, multi-colour-space input, making objects larger, copy-paste backgrounds, training on deployment conditions) and asked which were worth doing.

**What I kept vs. rewrote, and why:** Kept colour augmentation (grayscale and hue-shifted copies), because colour dependence was the one weakness that had been measured. Dropped multi-colour-space input (needs a modified network, loses pretrained weights) and copy-paste (aimed at a problem we had not observed).

**What the AI got wrong that I had to catch:**
- The first comparison (baseline 40 epochs vs augmented 20 epochs) showed a large cost on normal images (mallet AP50 0.938 vs 0.867). Both learning curves were still rising, so that comparison was on undertrained models. Rerun at 100 vs 50 epochs, the gap nearly closed (0.928 vs 0.922).
- I misread the evaluation table at first, thinking the original / gray / hue_shift rows were three training methods. They are three versions of the test images for one model. Worth saying because the table layout invites that mistake.

**How I verified it:** Every copied label file was compared with its original (0 differences in 790), and a sheet of 18 augmented images was checked by eye for box alignment. Both models were scored on the same 209 validation images in three colour conditions. I printed both result files in my terminal and compared them row by row.

## 4. Training runs and time estimates

**What I asked:** To run the longer training overnight, and how long things would take.

**What I kept vs. rewrote, and why:** Kept the equal-training-steps setup (100 epochs on 790 images vs 50 on 1,580) so the comparison is fair.

**What the AI got wrong that I had to catch:**
- It told me the remaining runs would take 30-45 minutes when the first baseline had taken about 10. I questioned it; it checked the logged per-epoch time (16.4 s) and corrected itself. It had misread a pause while my laptop slept as slow training.
- After pushing the weights it reported the push as done. Git had printed "Everything up-to-date" even though the push failed with HTTP 400. It caught this itself by comparing the commit on GitHub with the local one, then fixed it by raising `http.postBuffer` for this repo.

**How I verified it:** Per-epoch times read from `results.csv`. Remote and local commit hashes compared with `git ls-remote` after each push.

## 5. Error analysis

**What I asked:** To do the error analysis on the final model.

**What I kept vs. rewrote, and why:** Kept `analyze_errors.py` and the NMS change in `predict.py` (IoU 0.7 to 0.5), which removed duplicate boxes with no loss of recall.

**What the AI got wrong that I had to catch:**
- Before the analysis it expected small objects to be the main problem and planned a resolution experiment. The size breakdown showed the opposite: large close-ups are missed most (mallet 10 of 22 large vs 0 of 19 small). The resolution experiment was dropped.
- The NMS threshold was chosen on the validation set, which has no separate test set behind it. Stated as a limitation.
- The "label problems" finding is from looking at the image sheets and was not counted.

**How I verified it:** The counts add up (found + missed = labelled for each class). The fresh-clone run reproduced the same table. The mistakes are saved as image sheets in `results/errors/`.

## 6. Video test

**What I asked:** How to do the video test without a mallet (I had asked whether to use an online video), then to run the model on three videos I filmed.

**What I kept vs. rewrote, and why:** Filmed my own videos of a claw hammer and a shaker bottle, as the brief asks, and did not use online footage. Kept `video_test.py` and chose to commit the result sheets even though they show my room.

**What the AI got wrong that I had to catch:**
- Its first runs of the script produced no output because of a shell quoting mistake (a variable holding three filenames was passed as one argument in zsh). It found this from the traceback and added a clear error for unreadable videos.
- It had suggested the non-orange hammer would be a direct test of the colour fix. The result was more basic: neither model detects the hammer at all, so the test says more about object shape variety than about colour.

**How I verified it:** The video frames have no labels, so the table only counts frames with a box. Whether boxes are on the right object was checked by eye on the sheets in `results/video/`. The baseline model was run on the same frames for comparison.

## 7. Efficiency and hard example mining

**What I asked:** To do the model-efficiency stretch goal, then the hard-example one.

**What I kept vs. rewrote, and why:** Kept both scripts and their tables. Did not attempt quantization.

**What the AI got wrong that I had to catch:**
- The first timing table showed the GPU at 1.9 ms for every input size. That was a measurement error: the Apple GPU runs asynchronously and the library's timer does not wait for it. Replaced with wall-clock timing of the whole predict call.
- The script crashed on its first run because a library call returned nothing in this version; fixed by counting parameters directly.
- CPU time jumps more than expected between 320 and 416. It repeats, and the cause is unknown. Reported as is.

**How I verified it:** Timing was run three times; results agree within a few ms except at 416 and 512, so those are reported as ranges. Accuracy at 640 matches the evaluation table.

## 8. Reproducibility and write-up

**What I asked:** To draft the run scripts and both documents.

**What I kept vs. rewrote, and why:** Kept `setup.sh` and `run_all.sh`. The documents are AI drafts built from the result files.

**What the AI got wrong that I had to catch:** The clean-clone test showed result files contained an absolute path from my machine; the scripts now print paths relative to the repo.

**How I verified it:** Cloned the repo from GitHub into an empty folder, ran `setup.sh` and `run_all.sh` with my key, and got the same evaluation and error-analysis numbers as the committed files. A one-epoch training run in the clean copy also completed.
