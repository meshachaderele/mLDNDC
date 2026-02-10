## Pipeline Execution Steps
1. **Download datasets** 
   Go to https://zenodo.org:
   - Download all the files
   - Move them all to this directory `data_processing/data/`
   
2. **Update `pipeline.py`**  
   Open `pipeline.py` and modify/verify:  
   - Crop name (`"WIWH"`)  
   - Crop type (`"winter"`)  

3. **Run pipeline**  
   ```bash
   python pipeline.py
   ```

4. **Update `feature_engineering.py`**  
   Open `feature_engineering.py` and set/verify crop name ( `"WIWH"`)

5. **Run feature engineering**  
   ```bash
   python feature_engineering.py
   ```