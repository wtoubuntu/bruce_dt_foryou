# Bruce's Data Viz Tool 📊

Bruce's Data Viz Tool is a powerful, interactive Streamlit web application designed to help you quickly explore, compare, and visualize multiple datasets without writing a single line of code.

## 🚀 Features

- **Multi-File Upload Settings**: Easily upload multiple CSV or Excel files (`.csv`, `.xlsx`, `.xls`) to compare them directly against each other.
- **Smart Data Parsing**: Automatically detects common datetime columns and correctly parses them.
- **Support for Turbine Sensor Data**: Natively handles specialized turbine sensor CSV formats alongside standard CSV formats.
- **Advanced Downsampling**: Integrates the LTTB (Largest Triangle Three Buckets) downsampling algorithm to securely and cleanly visualize massive native datasets without crashing the browser or losing visual shape integrity.
- **Interactive Visualizations**:
  - **📈 Time Series**: Compare multiple files on the same time series graph. Includes resample controls, custom range bars, and interactive zooming.
  - **📊 Scatter Plot**: Analyze relationships between features across files, with built-in regression lines.
  - **📊 Statistics**: Discover underlying patterns with histograms, box plots, density plots, correlation heatmaps, and more.
  - **🌐 3D Visualization**: Explore multi-dimensional data relationships using 3D scatter plots.
- **Exporting Options**: Download your interactive Plotly graphs directly as standalone HTML files for sharing or presentations.

## 🛠 Prerequisites

To run this application, you will need Python 3 installed. You will also need to install the following Python packages:

- `streamlit`
- `pandas`
- `plotly`
- `numpy`
- `scipy`
- `openpyxl` (for reading `.xlsx` files)

You can install all dependencies via pip:
```bash
pip install streamlit pandas plotly numpy scipy openpyxl
```

## 🔧 How to Run

1. Clone or download this repository to your local machine.
2. Open your terminal or command prompt.
3. Navigate into the project folder.
4. Run the application using Streamlit:
```bash
streamlit run app.py
```
5. The application will immediately open in your default web browser (usually at `http://localhost:8501`).

## 📖 Usage Guide

1. **Upload Data**: Drag and drop your `.csv` or `.xlsx` files into the upload box on the main screen. You can select multiple files at once.
2. **Review Data**: Check the auto-generated Data Summary sidebar to make sure your columns and datetimes were loaded accurately.
3. **Visualize**: Use the tab navigation (`Time Series`, `Scatter Plot`, `Statistics`, `3D Visualization`) to swap between different visualization methods.
4. **Compare Files**: In the configuration menus for each plot, look for the `Select files to overlay` dropdown to pick which files to compare simultaneously. 

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! If you'd like to help improve the tool:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push -u origin feature/AmazingFeature`)
5. Open a Pull Request!
