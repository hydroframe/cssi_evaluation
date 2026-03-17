This workflow chart outlines the structure of the 2026 Modeling Hackathon. It shows how participants move from environment setup into the ParFlow and National Water Model (NWM) evaluation paths, and highlights how additional model workflows can be integrated into the same framework in future hackathons.

```mermaid
flowchart TD
    A[Participant enters workshop] --> B[Set up environment]
    B --> B1[HydroData login]
    B --> B2[GitHub access]
    B --> B3[Python environment]
    B --> B4[Test imports or Jupyter example]

    B4 --> C{Which model path?}

    C --> D[Day 1: ParFlow path]
    C --> E[Day 2: NWM path]
    C --> F[Future model path]

    D --> D1[Access observations through HydroData]
    D1 --> D2[Read ParFlow outputs with model-specific tools]
    D2 --> D3[Apply shared utilities]
    D3 --> D4[Run snow or streamflow evaluation]
    D4 --> D5[Review plots and metrics]

    E --> E1[Access reference data from external links]
    E1 --> E2[Read NWM outputs with model-specific tools]
    E2 --> E3[Apply shared utilities]
    E3 --> E4[Run variable-specific evaluation]
    E4 --> E5[Review plots and metrics]

    F --> F1[Create a new model-specific adapter]
    F1 --> F2[Match model outputs to framework format]
    F2 --> F3[Connect observations or reference data]
    F3 --> F4[Reuse shared metrics and plotting]
    F4 --> F5[Add variable-specific methods as needed]

    D5 --> G[Collect feedback]
    E5 --> G
    F5 --> G

    G --> G1[Challenges and incompatibilities]
    G --> G2[Integration opportunities]
    G --> G3[Needed data sources]
    G --> G4[Needed variables and metrics]

    classDef start fill:#d9edf7,stroke:#31708f,color:#111;
    classDef day1 fill:#e7f4e4,stroke:#3d7a3a,color:#111;
    classDef day2 fill:#fff1cc,stroke:#9b7a19,color:#111;
    classDef future fill:#f4e1f5,stroke:#8d4a8f,color:#111;
    classDef feedback fill:#f9e0e6,stroke:#9c3758,color:#111;

    class A,B,B1,B2,B3,B4,C start;
    class D,D1,D2,D3,D4,D5 day1;
    class E,E1,E2,E3,E4,E5 day2;
    class F,F1,F2,F3,F4,F5 future;
    class G,G1,G2,G3,G4 feedback;
```
