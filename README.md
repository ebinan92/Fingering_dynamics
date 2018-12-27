# Fingering_dynamics
two phase lattice boltzmann with zou-he boundary and half way bounce-back
[citation](https://www.sciencedirect.com/science/article/pii/S0377025715002037?via%3Dihub)

    ├── lattice_boltzmann            # Lattice boltzmann code
    │   ├── bounce_back.py           # bounce_back for rectangle or circle block 
    │   ├── create_block.py          # create circle or rectangle block
    │   ├── fingering.py             # main code
    │   ├── fingering_mei.py         # test for consistent initialization proposed by Renwei Mei (https://pdfs.semanticscholar.org/1767/c29ef4f09d1450f9ed69f4f2df6fb60ae809.pdf)
    │   └── valitation.py            # wettability validation
    |── spectrum_method              # viscous fingering simulation by spectrum_method
    │   ├── main.py                  # Non reactive viscous fingering (https://aip.scitation.org/doi/10.1063/1.866726)
    │   ├── main_3phase.py           # three phase precipitation viscous fingering (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.93.023103)
    │   
    
./lattice_boltzmann/fingering.py

![](ezgif.com-video-to-gif-compressor.gif)


./lattice_boltzmann/validation.py

![](wettability_validaton.png)

./spectrum_method/main_3phase.py

![](3phase_precipitationVF.git)
