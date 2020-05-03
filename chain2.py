class Chain:
    def __init__(self, 
            segment_count = 5, 
            segment_list = None,
            start_location = None,
            start_rotvec = None):

        if segment_list is None:
            self._segment_count = segment_count
            self._segments = [Segment() for idx in range(segment_count)]
        else:
            self._segment_count = len(segment_list)
            self._segments = [copy.deepcopy(segment) for segment in segment_list]

        if start_location is None:
            start_location = np.array([0,0,0])
        if start_rotvec is None:
            start_rotvec = np.array([0,0,0])

        self._start_location = start_location
        self._start_rotvec = start_rotvec

        self._segment_locations = [np.array([0,0,0]) for idx in range(segment_count)]
        self._segment_rotvecs = [np.array([0,0,0]) for idx in range(segment_count)]

# property getters and setters

    @property
    def segment_count(self):
        return self._segment_count

    #This is not intended as a way to modify segments
    @property
    def segments(self):
        return self._segments

    #Generic function for setting properties of any type of segment
    def SetSegmentProperties(idx,*args,**kwargs):
        self._segments[idx].SetProperties(*args,**kwargs)

        

