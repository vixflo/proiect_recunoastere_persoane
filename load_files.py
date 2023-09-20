import av
import torch
import numpy as np
import pickle
import lz4framed

# Funcția load_compressed_tensor  (Trebuie să ne asigurăm că gestionăm excepțiile în cazul în care apare o problemă în timpul deschiderii fișierului sau a decodificării)

def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval

# Clasa VideoReader: Aici ar trebui să ne sigurăm că tratăm cazurile în care fișierul nu există sau formatul nu este specificat.
class VideoReader(object):
    """
    Wrapper for PyAV
    Citește cadrele dintr-un fișier video în tensori de numpy.

    Exemplu:

    file = VideoReader(filename)
    video_frames = file[start_frame:end_frame]

    Dacă doriți, poate fi furnizat un fișier de tabel-de-conținut (TOC) pentru a accelera timpul de încărcare/căutare.
    """
    # Constructorul (__init__): pentru gestionarea resurselor, putem folosi un context manager (with) pentru a ne asigura că containerul este închis corect la final.

    def __init__(self, file, toc=None, format=None):
        if not hasattr(file, 'read'):
            file = str(file)
        self.file = file
        self.format = format
        self._container = None

        with av.open(self.file, format=self.format) as container:
            stream = [s for s in container.streams if s.type == 'video'][0]
            self.bit_rate = stream.bit_rate

            # Build a toc
            if toc is None:
                packet_lengths = []
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.stream.type == 'video':
                        decoded = packet.decode()
                        if len(decoded) > 0:
                            packet_lengths.append(len(decoded))
                            packet_ts.append(decoded[0].pts)
                self._toc = {
                    'lengths': packet_lengths,
                    'ts': packet_ts,
                }
            else:
                self._toc = toc

            self._toc_cumsum = np.cumsum(self.toc['lengths'])
            self._len = self._toc_cumsum[-1]

            # PyAV Returnează întotdeauna cadre în culori și facem asta
            # presupunere valoare în get_frame () mai târziu, deci acum am setat valoarea 3 pentru moment:
            self._im_sz = stream.height, stream.width, 3
            self._time_base = stream.time_base
            self.rate = stream.average_rate

        self._load_fresh_file()

#  Funcția _next_video_packet: Verificăm dacă există video în container pentru a evita erorile.
    @staticmethod
    def _next_video_packet(container_iter):
        for packet in container_iter:
            if packet.stream.type == 'video':
                decoded = packet.decode()
                if len(decoded) > 0:
                    return decoded

        raise ValueError("Nu am putut găsi pachete video.")
    
# Funcția _load_fresh_file: Ne asigurăm că închidem containerul anterior înainte de a-l redeschide pentru a evita scurgeri de resurse.
    def _load_fresh_file(self):
        if self._container is not None:
            self._container.close()

        if hasattr(self.file, 'seek'):
            self.file.seek(0)

        self._container = av.open(self.file, format=self.format)
        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        self._current_packet_no = 0

    @property
    def _video_stream(self):
        return [s for s in self._container.streams if s.type == 'video'][0]

    def __len__(self):
        return self._len

    def __del__(self):
        if self._container is not None:
            self._container.close()

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if item.start < 0 or item.start >= len(self):
            raise IndexError(f"start index ({item.start}) out of range")

        if item.stop < 0 or item.stop > len(self):
            raise IndexError(f"stop index ({item.stop}) out of range")

        return np.stack([self.get_frame(i) for i in range(item.start, item.stop)])

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def toc(self):
        return self._toc

# Funcția get_frame: Aici ne asigurăm că se caută pachetul corespunzător și se extrage cadrul dorit.
# Trebuie să ne asigurăm că se gestionează corect situația în care valoarea j este în afara intervalului valid.
    def get_frame(self, j):
        # Găsiți pachetul în care se află acest cadru.
        packet_no = self._toc_cumsum.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Găsiți locația cadrului în pachet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc_cumsum[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame

        return frame.to_ndarray(format='rgb24')

# Funcția _seek_packet: Să ne sigurăm că gestionăm corect cazurile în care nu se găsește pachetul dorit.
    def _seek_packet(self, packet_no):
        """ Avansați prin generatorul de containere până când obținem pachetul
        dorit de noi. Stocați pachetul respectiv selfpp._current_packet."""
        packet_ts = self.toc['ts'][packet_no]
        # Căutați doar atunci când este nevoie.
        if packet_no == self._current_packet_no:
            return
        elif (packet_no < self._current_packet_no
            or packet_no > self._current_packet_no + 1):
            self._container.seek(packet_ts, stream=self._video_stream)

        demux = self._container.demux(self._video_stream)
        self._current_packet = self._next_video_packet(demux)
        while self._current_packet[0].pts < packet_ts:
            self._current_packet = self._next_video_packet(demux)

        self._current_packet_no = packet_no