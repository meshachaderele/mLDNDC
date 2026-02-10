import pandas as pd


def add_precipitation_temperature(CROP, CROP_TYPE):
    df = pd.read_csv("data/climate_10km_2010_2020.csv")
    df_class_climate = pd.read_csv("data/climate_ai_classified.csv")
    sim_ids_df = pd.read_parquet(f"data/simulation_ids_{CROP}.parquet")
    
    
    
    # Create the new dataframe with selected columns
    
    
    def build_climate_features(clim: pd.DataFrame, crop_type: str) -> pd.DataFrame:
        """
        Build seasonal climate features for winter or spring crops.
    
        Parameters
        ----------
        clim : pd.DataFrame
            Must contain columns: id, date, tavg, prec.
            'date' can be string or datetime; will be converted.
        crop_type : {"winter", "spring"}
            Type of crop determining the growing season definition.
    
        Returns
        -------
        pd.DataFrame
            Aggregated features by id and harvest_year.
        """
        if crop_type not in {"winter", "spring"}:
            raise ValueError("crop_type must be 'winter' or 'spring'")
    
        df = clim.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    
        if crop_type == "winter":
            # Growing season for harvest year Y: Oct (Y-1) – Jul (Y)
            df["harvest_year"] = df["year"]
            df.loc[df["month"].isin([10, 11, 12]), "harvest_year"] = df["year"] + 1
    
            # Seasonal flags
            df["is_autumn"] = df["month"].isin([10, 11])
            df["is_winter"] = df["month"].isin([12, 1, 2])
            df["is_spring"] = df["month"].isin([3, 4, 5])
            df["is_gs"] = df["month"].isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7])
    
            # Yearly totals (by harvest_year)
            yearly = df.groupby(["id", "harvest_year"], as_index=True).agg(
                total_precipitation_year=("prec", "sum"),
                total_average_temperature_year=("tavg", "sum"),
            )
    
            # Growing season
            gs = (
                df[df["is_gs"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_growing_season=("prec", "sum"),
                    total_average_temperature_growing_season=("tavg", "sum"),
                )
            )
    
            # Autumn: Oct, Nov (previous year)
            autumn = (
                df[df["is_autumn"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_autumn=("prec", "sum"),
                    total_average_temperature_autumn=("tavg", "sum"),
                )
            )
    
            # Winter: Dec (prev), Jan, Feb (current)
            winter = (
                df[df["is_winter"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_winter=("prec", "sum"),
                    total_average_temperature_winter=("tavg", "sum"),
                )
            )
    
            # Spring: Mar, Apr, May
            spring = (
                df[df["is_spring"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_spring=("prec", "sum"),
                    total_average_temperature_spring=("tavg", "sum"),
                )
            )
    
            out = (
                yearly.join(gs, how="left")
                .join(autumn, how="left")
                .join(winter, how="left")
                .join(spring, how="left")
            )
    
        else:  # crop_type == "spring"
            # For spring crops, assume harvest_year = calendar year
            df["harvest_year"] = df["year"]
    
            # Seasonal flags
            df["is_spring"] = df["month"].isin([3, 4, 5])
            df["is_summer"] = df["month"].isin([6, 7, 8])
            df["is_gs"] = df["month"].isin([3, 4, 5, 6, 7, 8])
    
            # Yearly totals (by harvest_year)
            yearly = df.groupby(["id", "harvest_year"], as_index=True).agg(
                total_precipitation_year=("prec", "sum"),
                total_average_temperature_year=("tavg", "sum"),
            )
    
            # Growing season: Mar–Aug of harvest_year
            gs = (
                df[df["is_gs"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_growing_season=("prec", "sum"),
                    total_average_temperature_growing_season=("tavg", "sum"),
                )
            )
    
            # Spring: Mar–May
            spring = (
                df[df["is_spring"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_spring=("prec", "sum"),
                    total_average_temperature_spring=("tavg", "sum"),
                )
            )
    
            # Summer: Jun–Aug
            summer = (
                df[df["is_summer"]]
                .groupby(["id", "harvest_year"], as_index=True)
                .agg(
                    total_precipitation_summer=("prec", "sum"),
                    total_average_temperature_summer=("tavg", "sum"),
                )
            )
    
            out = (
                yearly.join(gs, how="left")
                .join(spring, how="left")
                .join(summer, how="left")
            )
    
        out = out.reset_index()
        return out
    
    
    new_df = build_climate_features(df, CROP_TYPE)
    new_df = new_df[~new_df['harvest_year'].isin([2010, 2021])]
    
    def assign_class(value):
        if 28 <= value <= 31.05:
            return 1
        elif 31.06 <= value <= 35:
            return 2
        elif 35 < value <= 45:
            return 3
        elif 45 < value <= 55:
            return 4
        else:
            return 'Unknown'  # For values outside the defined ranges
        
    df_class_climate['Class'] = df_class_climate['Aridity index'].apply(assign_class)
    
    climate_df = pd.merge(df_class_climate, new_df, on='id')
    
    climate_df = climate_df.drop(['latitude',	'longitude',	'temperature average', 'annual precipitation',	'Aridity index'], axis=1)
    
    sim_ids_df['climate_id'] = sim_ids_df['climate_id'].astype(int)
    
    sim_ids_climate = sim_ids_df.merge(
        climate_df,
        left_on="climate_id",
        right_on="id",
        how="left"
    )
    
    sim_ids_climate = sim_ids_climate.drop(['folder', 'id'], axis=1)
    
    sim_ids_climate.to_parquet(f"data/simulation_ids_{CROP}_with_climate.parquet", index=False)







































