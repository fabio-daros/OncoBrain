import os
from synapseclient import Synapse
from synapse_sync_util import sync_from_synapse

DEST_DIR = "dataset/bmt"
SYNAPSE_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc0OTA4MjM1OSwiaWF0IjoxNzQ5MDgyMzU5LCJqdGkiOiIyMTI3OSIsInN1YiI6IjM1NDU2NTAifQ.Lvp1BJO8wCfNp7q5MbHSQYemvAp0Pbt6sGzUmHCL3NUOhiuAvG35bTR72sk190-t8vY2pYqTL7h655deq1JyaRLS_A_J0HSPe-t1nnqGsd_Awsexcz8v85302GuZtFOOaA2fJcp6aFPAXC2rTuQm0cxXmDiC7SF97JrQUaWQgr0V5JziwhVybsJNyg3Vw1gHzBKXThXKHBLa4NRJTwJ4ygRorPM7DyvJaAOM3Ms02XuuS1jV1vmUqHV0EsJJ1vr21wkXyBV3yCRdDED2YLsHaZcKeYANMaGibukZlJnsiPVWmre3ulxbhHWVlEiYjvA_W7ulJi8T2T3i4hXAd26jnA"
SYNAPSE_PROJECT_ID = "syn55259257"

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    syn = Synapse()
    syn.login(authToken=SYNAPSE_TOKEN)

    print("Starting the BMT DataSet Download...")
    sync_from_synapse(syn, SYNAPSE_PROJECT_ID, DEST_DIR)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
