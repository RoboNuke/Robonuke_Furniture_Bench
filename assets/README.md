# assets

One folder per FurnitureBench part, named after the part as FurnitureBench names it
(`square_table_top`, `square_table_leg1` … `square_table_leg4`, `drawer_box`, …):

    assets/
      square_table_top/
        square_table_top.usd
        metadata.yaml
      square_table_leg1/
        ...

Each folder holds the part's `.usd` and a `metadata.yaml`. The env resolves parts from here;
no environment variable is needed once the files are in place.
