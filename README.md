# Problem:

When an earthquake occurs, the initial disaster is just the beginning. As stated in an article by the
United States Geological Survey, “Earthquakes don’t kill people, buildings and their contents
do”. Aftershocks, broken foundations, and weak structures can cause buildings to collapse even
after the worst has passed. But with lives on the line, rescue workers need to act quickly while
also keeping themselves and those they're trying to save safe. There needs to be a way to quickly
check if an area is safe to enter for first responders before arriving on the scene. Or to inform the
public to stay clear of areas where there is risk of a building, street, or other structure collapsing
imminently.

# Soultion:

We created a simulation that can generate a city after the event of an earthquake. Using two models imported from Blender to represent the destruction.
One was a collapsed/broken building and the other was an uneffected building. In this environment we created a trajectory for a drone that would fly above the city and take pictures of the
buildings below it. These images are then processed by a model we trained on images of the two building types to determine whether the building was damaged during the 
earthquake. That prediction is then sent to a server running locally on our computer representing a computer on the ground that compiles the predictions to create a heatmap of the city. With black representing safe areas and white representing potential danger zones. 

# How to Execute Code:

## Requirements
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Anaconda Navigator](https://www.anaconda.com/docs/getting-started/anaconda/install)

## 1. Download and Install Conda or Anaconda

Above you can find the link to the getting started guide for both Conda (CLI) or Anaconda (GUI). If you dont have either installed follow the tutorial to download it for your respective Operating System before continuing.

## 2. Clone the repository

`git clone --recurse-submodules https://github.com/linktimemaster/DisasterDrone.git`

## 3. Import and Activate Conda Environment

```bash
cd DisasterDrone
conda env create -f drones.yml
conda activate drones
```

## 4. Run Server and Simulation
Open two terminals, on the first run the server.py and on the second run the 576Project.py (the simulation). Make sure both files are running through in the **drones** conda environment.
### If done correctly your screen should look something like this:

<img width="3051" height="1978" alt="image" src="https://github.com/user-attachments/assets/a469c1d8-ee5b-408a-a1d4-b084285d95a9" />

### Output Heatmap:

- White: Collaplsed
- Black: Still Standing

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/d499a73e-6715-42c3-b274-6057a7ff2558" />
