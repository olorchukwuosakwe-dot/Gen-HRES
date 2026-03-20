# Gen-HRES Project

This guide provides instructions on how to set up and run the project using a Python virtual environment (`venv`).

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

## Setup Instructions

### 1. Clone the Repository
Open your terminal and navigate to the project directory:
```bash
cd Gen-HRES
```

### 2. Create a Virtual Environment
Create a new virtual environment named `venv` in the project folder:
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
Activate the environment to ensure dependencies are installed locally:
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

### 4. Install Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Running the Project

Once the environment is activated and dependencies are installed, you can run the main script:
```bash
python3 fetch_nasa_power.py && python3 preprocess_nasa_data.py && python3 train_cgan.py
```

## Deactivating the Environment
When you are finished working, you can exit the virtual environment by typing:
```bash
deactivate
```

## Troubleshooting
If you encounter any issues with package versions, try updating `pip` or checking the `requirements.txt` for compatibility.
