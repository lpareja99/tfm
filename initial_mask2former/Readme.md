1. Build img `docker compose up -d --build` (~10min)

2. Start container witout building `docker compose up -d`

3. Access Container `docker exec -it road_defect_tfm bash`

4. Train data `mim train mmseg configs/initial_test_flowity.py`

5. Test Data 