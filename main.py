# read the string filename
filename = input()

def extract_timestamps(line):
    i = line.find('[')
    if i == -1:
        return None
    
    j = line.find(']', i+1)
    
    if j == -1:
        return None
    
    ts_full = line[i+1:j]
    # Strip timezone (e.g., "-0400"); keep only up to seconds
    return ts_full.split(' ')[0]

def write_multi_timestamps(infile):
    counts = {}
    with open(infile, 'r') as f:
        for line in f:
            ts = extract_timestamps(line)
            if ts:
                counts[ts] = counts.get(ts, 0) + 1
    
    outname = f"req_{infile}"
    with open(outname, 'w') as out:
        for ts in sorted(k for k, v in counts.items() if v > 1):
            out.write(ts + "\n")
            
write_multi_timestamps(filename)