def ctc_greedy_decoder(logits, idx_to_char, blank=0):
    ids = logits.argmax(dim=2).transpose(0,1)  # (B,T)
    out = []
    for seq in ids:
        prev = blank
        chars = []
        for i in seq.tolist():
            if i!=prev and i!=blank:
                chars.append(idx_to_char.get(i,""))
            prev = i
        out.append("".join(chars))
    return out

def edit_distance(a, b):
    n,m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def cer(preds, tgts):
    te = sum(edit_distance(p,t) for p,t in zip(preds,tgts))
    tc = sum(len(t) for t in tgts)
    return te/tc if tc>0 else 0.0

def wer(preds, tgts):
    te,tw = 0,0
    for p,t in zip(preds,tgts):
        pw,twl = p.split(), t.split()
        te += edit_distance(pw,twl)
        tw += len(twl)
    return te/tw if tw>0 else 0.0
