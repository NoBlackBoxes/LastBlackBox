# Crick : Open Projects
Day five: you have a full-stack robot. Today is about using it. Pick a track in the morning, scope a project, build and demo by 16:30.

A few advanced topics that the tracks build on are listed below. Pick the ones relevant to your project.

## Topics
{Audio:Signal-Processing}
{Vision:Image-Processing}
{Intelligence:Large-Language-Models}

# The tracks

Pick one. The starter for each track is already on your Pi.

**Track A: longitudinal sensor streaming.** Stream any sensor on the robot to a live plot on your laptop. Starter scripts live in `enb/07_microphones-and-speakers/_resources/`. Project ideas: stream LDR plus temperature plus sound-level together; characterise latency and packet loss; swap in a sensor of your own.

**Track B: audio signal processing.** Mics in, FFT or filter in the middle, speaker out. Starter scripts live in `boxes/audio/`. Project ideas: real-time pitch shifter, clap detector, ambient-sound classifier.

**Track C: computer vision.** Live camera frames in, OpenCV in the middle, behaviour out. Starter scripts live in `enb/10_computer-vision/_resources/`. Project ideas: face-following robot, colour tracker, motion-triggered photo capture.

**Track D: voice interaction.** Speech-to-text plus an LLM plus text-to-speech. Starter scripts live in `course/versions/ai-workshops/01_audio/_resources/python/`. Project ideas: an NB3 that answers questions, a robot with a custom persona, a story-teller.

**Track E: full stack.** Combine yesterday's web GUI with today's senses. Project ideas: drive the robot from your phone while seeing its camera; teleoperated audio chat; mobile dashboard for live sensor plots.

# Build and demo

**MVP first, polish later.** Get the smallest thing that works, then iterate. Commit often, on a feature branch. When in doubt, `print()` everything; clean up at the end. Ask the TAs early.

**Before the demo,** strip debug prints, add a one-line banner so the audience knows what is starting, and practise the demo once so the live run does not surprise you. Have a screenshot or short video as a fallback.

**Three minutes each at 16:00.** Say what you built (one sentence), how it works (one sentence), run a 60-second live demo, say what you would do next.
