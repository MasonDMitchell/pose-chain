class Chain:
    def __init__(self, 
            segment_count = 5, 
            segment_list = None,
            start_location = None,
            start_orientation = None):

        if segment_list is None:
            self._segment_count = segment_count
            self._segments = [Segment() for idx in range(segment_count)]
        else:
            self._segment_count = len(segment_list)
            self._segments = [copy.deepcopy(segment) for segment in segment_list]

        if start_location is None:
            start_location = np.array([0,0,0])
        if start_orientation is None:
            start_orientation = np.array([0,0,0])

        self._start_location = start_location
        self._start_orientation = start_orientation

        self._UpdateCalculatedProperties()

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

# calculated getters and related functions

    @property
    def segment_locations(self):
        return self._segment_locations

    @property
    def segment_orientations(self):
        return self.segment_orientations

    def _UpdateCalculatedProperties(self):
        #update orientations must be called before update locations
        self._UpdateSegmentOrientations()
        self._UpdateSegmentLocations()

    def _UpdateSegmentLocations(self):
        final_orientations = self.final_orientations
        final_locations = [self._start_location]

        for segment_idx in range(self.segment_count):
            final_locations.append(
                    self._segments[segment_idx].final_location)

    def _UpdateSegmentOrientations(self):
        final_orientations = [self._start_location]

        for segment_idx in range(self.segment_count):
            final_orientations.append(
                    self._segments[segment_idx].final_orientation)

# other functions
    
    # t_array is an array of floats from 0 to segment_count
    def GetPoints(self, t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        raw_segment_points = []

        t_array = np.sort(t_array)
        for segment_idx in range(self.segment_count):

            segment_t = t_array[np.logical_and(
                segment_idx <= t_array,
                segment_idx + 1 > t_array)]

            raw_segment_points.append(
                    self._segments[segment_idx].GetPoints(segment_t))
        

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        
