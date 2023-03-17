import os
import sqlite3
import pandas as pd
import numpy
import os

def main():
    df = pd.DataFrame()
    experiment_dirs = ["10", "15", "20", "25"]
    nr_of_runs = 5
    csv_path = "data.csv"

    #Main dir
    for dir_path in experiment_dirs:
        #Experiment_dir
        if os.path.exists(dir_path) == False:
            continue

        os.chdir(dir_path)

        for run_nr in range(0,nr_of_runs):
            # Run_dir
            os.chdir("Run " + str(run_nr))

            database = "database0" + "/db.sqlite"
            conn = sqlite3.connect(database)
            df_temp = pd.read_sql(
                "SELECT ea_optimizer_individual.individual_id, ea_optimizer_generation.generation_index, Distance.value AS Distance, Power.value AS Power, Joints.value AS Joints\n" +
                "FROM ea_optimizer_generation\n" +
                "JOIN ea_optimizer_individual ON ea_optimizer_generation.individual_id = ea_optimizer_individual.individual_id\n" +
                "JOIN float AS Distance ON ea_optimizer_individual.distance_obj_id = Distance.id\n" +
                "JOIN float AS Power ON ea_optimizer_individual.remaining_power_id = Power.id\n" +
                "JOIN float AS Joints ON ea_optimizer_individual.number_of_joints_id = Joints.id\n", con=conn)

            max_parts_col = [int(dir_path) for i in range(0, len(df_temp.index))]
            df_temp["Max parts"] = max_parts_col

            run_col = [run_nr for i in range(0, len(df_temp.index))]
            df_temp["Run number"] = run_col

            df = pd.concat([df, df_temp])
            df.reset_index(inplace=True, drop=True)
            #Go back to Experiment_dir
            os.chdir("..")
        #Go back to Main_dir
        os.chdir("..")

        if os.path.exists(csv_path):
            os.remove(csv_path)

        df.to_csv(csv_path)

        print(df)
if __name__ == "__main__":
    main()