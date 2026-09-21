\begin{table*}[t]
  \centering
  \caption{Comparison under the TangoFlux AudioCaps-886 protocol.  $^{\ddagger}$ Results are directly taken from the TangoFlux benchmark \cite{hung2024tangoflux}; $^{\dagger}$ denotes our evaluation of official checkpoints. Bold denotes the best performance in each metric column. The FD$_{\text{OpenL3}}$ 16k ref.\ column is computed bandwidth-fair: both the generated audio and the reference set are downsampled to 16\,kHz and the first 10\,s is used, so no model is penalized for its native sampling rate (44.1\,kHz TangoFlux / 24\,kHz EzAudio would otherwise be inflated against the 16\,kHz references).}
  \label{tab_01}
  \vspace{0.5em}

  \fontsize{8.5pt}{9.5pt}\selectfont
  \renewcommand{\arraystretch}{1.0}
  % \setlength{\tabcolsep}{3.0pt}
  \begin{tabular}{l c c c c c c c c c}
  
    \toprule
    Model & Params. & Text Cond. & Mel Vocoder & NFE
    & FD$_{\text{OpenL3}}$ full $\downarrow$
    & FD$_{\text{OpenL3}}$ 16k ref. $\downarrow$
    & KL$_{\text{PaSST}}$ $\downarrow$
    & CLAP $\uparrow$
    & IS $\uparrow$ \\
    \midrule
    % \multicolumn{10}{l}{\textit{Published TangoFlux benchmark}} \\
    AudioLDM2-Large $^{\ddagger}$ \cite{liu2023audioldm2}     & 712M  & T5 + CLAP & $\checkmark$ & 200 & 72.4735 & 40.0246 & 1.810 & 0.419 & 7.90 \\
    Stable Audio Open $^{\ddagger}$ \cite{evans2024stableaudioopen} & 1056M & T5             & $\times$     & 100 & 89.20  & N/A (paper only) & 2.580 & 0.291 & 9.90 \\
    Stable Audio Open (official ckpt, 10s) $^{\dagger}$ & 1056M & T5             & $\times$     & 100 & 194.6241 & 74.0086 & 2.232 & 0.306 & 10.17 \\
    Tango2 $^{\ddagger}$ \cite{majumder2024tango2}            & 866M  & T5        & $\checkmark$ & 200 & 88.3099 & 45.4501 & 1.110 & 0.447 & 9.00 \\
    TangoFlux-Base \cite{hung2024tangoflux} $^{\ddagger}$    & 515M  & T5        & $\times$     & 50  & 80.20  & N/A (paper only) & 1.220 & 0.431 & 11.70 \\
    TangoFlux-Base (official ckpt, 30s) $^{\dagger}$          & 515M  & T5        & $\times$     & 50  & 179.3782 & 49.7599 & N/A & N/A & N/A \\
    TangoFlux-Base (official ckpt, 10s clean-mask stage1) $^{\dagger}$ & 515M  & T5        & $\times$     & 32  & 82.3786 & 46.4879 & 1.8706 & 0.4345 & 8.6032 \\
    TangoFlux-RL $^{\ddagger}$ \cite{hung2024tangoflux}         & 515M  & T5        & $\times$     & 50  & 75.10  & N/A (paper only) & 1.150 & 0.480 & \textbf{12.20} \\
    TangoFlux-RL (official ckpt, 30s) $^{\dagger}$             & 515M  & T5        & $\times$     & 50  & 167.1808 & 52.2807 & N/A & N/A & N/A \\
    MeanAudio $^{\dagger}$ \cite{li2025meanaudio}        & 120M & T5 + CLAP & $\checkmark$ & 25
      & 121.79 & 88.1107 & 1.214 & 0.455 & 10.58 \\
    EzAudio $^{\dagger}$   \cite{hai2024ezaudio}      & 875M & T5 + CLAP & $\times$ & 50
      & \textbf{38.26} & 41.8930 & 1.179 & 0.496 & 9.77 \\
    GenAU$ ^{\dagger}$    \cite{haji2026taming}       & 1250M & T5 + CLAP & $\checkmark$ & 200
      & 68.92 & 31.9406 & 1.379 & 0.473 & 10.38 \\
    \midrule
    \multicolumn{10}{l}{\textit{Unite-Audio (ours)}} \\
    Unite-Audio
      & \textbf{117M} & T5 & $\times$ & 32 & 84.76 & 43.2529 & 1.107
      & \textbf{0.534} & 11.34 \\
    Unite-Audio
      & \textbf{117M} & T5 & $\times$ & 16 & 83.22 & 39.3843
      & \textbf{1.057}
       & 0.532 & 11.04 \\
    Unite-Audio
      & \textbf{117M} & T5 & $\times$ & 8 & 84.88 & 42.4136 & 1.079
       & 0.530 & 10.85 \\
    Unite-Audio
      & \textbf{117M} & T5 & $\times$ & 4 & 88.91 & 48.8636 & 1.175
       & 0.512 & 9.73 \\
    \bottomrule
  \end{tabular}
  \\
\end{table*}
