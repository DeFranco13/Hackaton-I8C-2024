import { Component } from '@angular/core';
import { ApiService } from '../service/api.service';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent {
    loading= true
    anomalies: any[] = []

  constructor(private apiservice: ApiService){}

  ngOnInit(){
    this.apiservice.getData().subscribe({
      next: (data) => {
        this.anomalies = data;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error fetching data', error);
        this.loading = false;
      }
    });
  }
}
