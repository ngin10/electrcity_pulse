# Databricks notebook source
# MAGIC %md
# MAGIC # Unpack an uploaded landing zip into the landing volume (the no-CLI way to load data)
# MAGIC
# MAGIC The Databricks UI uploads files, not folder trees. So:
# MAGIC 1. On your machine, zip the landing folder, from the repo root (PowerShell):
# MAGIC    `Compress-Archive -Path data\landing\* -DestinationPath landing.zip -Force`
# MAGIC 2. Catalog > pulse > bronze > Volumes > landing > **Upload to this volume**: upload `landing.zip`.
# MAGIC 3. Run this notebook. It extracts every file not already in the volume, keeping the folder layout,
# MAGIC    then deletes the zip.
# MAGIC
# MAGIC Files already in the volume are never overwritten, so you can upload a zip of the whole landing folder
# MAGIC again later and only the new files are added. Auto Loader then loads each new file once.

# COMMAND ----------

import os
import zipfile

dbutils.widgets.text('catalog', 'pulse')  # noqa: F821
VOLUME = f"/Volumes/{dbutils.widgets.get('catalog')}/bronze/landing"  # noqa: F821
ZIP = f'{VOLUME}/landing.zip'

added = skipped = 0
with zipfile.ZipFile(ZIP) as archive:
    for entry in archive.infolist():
        # Windows PowerShell 5.1 writes backslashes in zip entry names
        name = entry.filename.replace('\\', '/').lstrip('/')
        if entry.is_dir() or not name.endswith('.parquet'):
            continue
        target = f'{VOLUME}/{name}'
        if os.path.exists(target):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Write to a temporary name and rename, so Auto Loader never sees a half-written file
        with archive.open(entry) as source, open(target + '.tmp', 'wb') as out:
            out.write(source.read())
        os.rename(target + '.tmp', target)
        added += 1

os.remove(ZIP)
print(f'{added} files added, {skipped} already there')

# COMMAND ----------

# The 15 landing folders and their file counts
for folder in sorted(os.listdir(VOLUME)):
    count = sum(len([f for f in files if f.endswith('.parquet')]) for _, _, files in os.walk(f'{VOLUME}/{folder}'))
    print(f'{folder:32s} {count}')
