# Egyptian license Plate Recognition

The purpose of this project is to develop a robust and accurate Car License Plate Recognition (LPR) system. The system will be designed to automatically detect and recognize license plates from images, providing valuable applications in security, law enforcement, parking management, and more, we will mainly focus on the parking management application for this project.

## Table of Contents
- [Libraries Used](#libraries-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Libraries Used

List the libraries used in your project and their purposes:

- **NumPy**: Used for numerical operations and array manipulation in Python.
- **OpenCV (cv2)**: Employed for computer vision tasks like image processing and manipulation.
- **Math**: Standard Python library for mathematical operations.
- **scikit-learn (sklearn)**: Used for machine learning tasks, including the K-Nearest Neighbors (KNN) classifier.
- **Numba**: Utilized for JIT (Just-In-Time) compilation to optimize performance in specific functions.
- **scikit-image**: Leveraged for image processing tasks, including contour finding and drawing shapes on images.
- **PySimpleGUI**: Employed for creating a graphical user interface (GUI) for the application.
- **PIL (Python Imaging Library)**: Used for opening, manipulating, and saving various image file formats.
- **Other libraries used in your project**

## Installation

Ensure you have Python installed on your system. Then, run the following commands in your terminal to install the required libraries:

```bash
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install numba
pip install scikit-image
pip install PySimpleGUI
```

## Usage

### Running the Application

To run the application:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Ensure you have Python installed.
4. Install the necessary dependencies (as mentioned in the Installation section).
5. Run the main script to start the application.

```bash
python UI.py
```

## Using the Application

Upon launching the application, follow these steps:

1. **Launch the Application**: Execute `UI.py` to start the application.
2. **Image Selection**: The GUI will appear, presenting options to select an image.
3. **Select Image**: Click on the 'Browse' button to choose an image file containing a vehicle and its license plate.
4. **Image Processing**: The application will process the image, perform license plate recognition, and display the result.
5. **Result Display**: The recognized license plate characters will be presented in the 'Processing Result' section of the GUI.
6. **LED Indicator**: An LED indicator will show 'Allowed' or 'Banned' based on a predefined comparison array.

## Contributers
<table>
  <tr>
<td align="center">
<a href="https://github.com/MoSobhy01" target="_black">
<img src="https://avatars.githubusercontent.com/u/94849658?v=4" width="150px;" alt="Mahmoud Sobhy"/><br /><sub><b>Mahmoud Sobhy</b></sub></a><br />
</td>

<td align="center">
<a href="https://github.com/MhmoudYahia" target="_black">
<img src="https://avatars.githubusercontent.com/u/94763036?v=4" width="150px;" alt="Mahmoud Yehia"/><br /><sub><b>Mahmoud Yehia</b></sub></a><br />
</td>

<td align="center">
<a href="https://github.com/Yousef-Rabia" target="_black">
<img src="https://avatars.githubusercontent.com/u/78663127?v=4" width="150px;" alt="Youssef Rabia"/><br /><sub><b>Yousef Rabia</b></sub></a><br />
</td>

<td align="center">
<a href="https://github.com/Youssef-Hagag" target="_black">
<img src="https://avatars.githubusercontent.com/u/94843229?v=4" width="150px;" alt="Youssef Hagag"/><br /><sub><b>Youssef Hagag</b></sub></a><br />
</td>

</tr>
 </table>
